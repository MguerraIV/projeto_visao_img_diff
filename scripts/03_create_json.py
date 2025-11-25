import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig
from PIL import Image
import json
import os

class DifferenceCaptionGenerator:
    def __init__(self, model_id="llava-hf/llava-v1.6-mistral-7b-hf", load_in_4bit=True, use_fast=True):
        """
        Inicializa o modelo LLaVA.
        4-bit quantization para caber em GPUs comuns (como T4 ou RTX 3060).
        """
        print(f"Carregando VLM ({model_id})...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.processor = LlavaNextProcessor.from_pretrained(model_id, use_fast=use_fast)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, 
            quantization_config=quantization_config, 
            device_map="auto"
        )
        print("VLM Carregado!")

    def generate_description(self, image, prompt_text="Describe this image concisely."):
        """
        Gera uma descrição curta para um recorte de imagem.
        (Stage 1 do Paper: Object Labeling)
        """
        # Formato de prompt do LLaVA-NeXT/Mistral
        prompt = f"[INST] <image>\n{prompt_text} [/INST]"

        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.model.device)
        
        output = self.model.generate(
            **inputs, 
            max_new_tokens=50,
            do_sample=False 
        )
        
        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        
        if "[/INST]" in decoded:
            response = decoded.split("[/INST]")[-1].strip()
        else:
            response = decoded.strip()

        return response

    def create_dataset_entry(self, img_path_a, img_path_b, bbox):
        """
        Processa um par de imagens e uma bbox para criar a entrada final JSON.
        """
        img_a = Image.open(img_path_a).convert("RGB")
        img_b = Image.open(img_path_b).convert("RGB")
        
        # 1. Recortar as regiões (Crops)
        x1, y1, x2, y2 = map(int, bbox)
        crop_a = img_a.crop((x1, y1, x2, y2))
        crop_b = img_b.crop((x1, y1, x2, y2))
        
        # 2. Gerar legendas dos conteúdos (Stage 1 do paper)
        print(" -> Gerando legenda para a imagem esquerda...")
        caption_1 = self.generate_description(crop_a, "Describe the main object in this image concisely.")
        
        print(" -> Gerando legenda para a imagem direita...")
        caption_2 = self.generate_description(crop_b, "Describe the main object in this image concisely.")
        
        # 3. Gerar a legenda da diferença (Stage 2 do paper)
        total_width = img_a.width + img_b.width + 20
        max_height = max(img_a.height, img_b.height)
        
        concat_img = Image.new('RGB', (total_width, max_height), (0, 0, 0))
        concat_img.paste(img_a, (0, 0))
        concat_img.paste(img_b, (img_a.width + 20, 0))
        
        # Alimentando o VLM com o contexto visual e as legendas textuais
        diff_prompt = (
            f"Analyse the left image and the right image (separated by the black vertical bar). What is the difference between the red bounding box area in each image?"
            "Answer the question in a few concise sentences"
        )

        print(" -> Gerando a explicação final da diferença via VLM...")
        final_response = self.generate_description(concat_img, diff_prompt)

        # 4. Montar o JSON final
        # Normalizar bbox para 0-1 (formato padrão LLaVA/Visual Genome)
        w, h = img_a.size
        norm_bbox = [
            round(x1/w, 2), round(y1/h, 2), 
            round(x2/w, 2), round(y2/h, 2)
        ]
        
        entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": "Analyse the left image and the right image (separated by the black vertical bar). What is the difference between the red bounding box area in each image? Answer the question in a few concise sentences"
                },
                {
                    "from": "llm",
                    "value": final_response
                }
            ],
            "bbox": norm_bbox,
            "captions1": caption_1,
            "captions2": caption_2,
            "image_left": img_path_a,
            "image_right": img_path_b
        }
        
        return entry

if __name__ == "__main__":
    base_dir = "data/raw_captions"
    detected_data = os.listdir(base_dir)
    
    if not os.path.exists("data/caption_images"):
        print("Aviso: Imagens de exemplo não encontradas. O script vai falhar se não houver imagens reais.")

    try:
        generator = DifferenceCaptionGenerator(load_in_4bit=True, use_fast=True)
        final_dataset = []

        for cont, json_name in enumerate(detected_data):
            print(f"Processando par {cont}...")

            json_path = os.path.join(base_dir, json_name)
            json_path = json_path.replace("\\", "/")

            with open(json_path, "r") as f:
                data = json.load(f)

            # Verifica se arquivo existe antes de processar
            if os.path.exists(data['img_a']):
                entry = generator.create_dataset_entry(
                    data['img_a'], 
                    data['img_b'], 
                    data['bbox']
                )
                final_dataset.append(entry)
                print("Entrada criada com sucesso!")
            else:
                print("Imagem não encontrada, pulando...")

        os.makedirs("data/final_dataset", exist_ok=True)
        with open("data/final_dataset/img_diff_train.json", "w") as f:
            json.dump(final_dataset, f, indent=2)
            
        print("\nDataset Final Gerado em 'data/final_dataset/img_diff_train.json'")
        
    except Exception as e:
        print(f"\nErro ao executar o modelo: {e}")