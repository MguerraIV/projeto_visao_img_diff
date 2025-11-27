import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
import json

class ImagePairGenerator:
    def __init__(self, output_dir="data/generated_images", device="cuda"):
        """
        Inicializa o gerador usando SDXL.
        Referência: O paper utiliza Stable-Diffusion-XL para alta qualidade.
        """
        self.output_dir = output_dir
        self.device = device
        
        # Cria diretórios se não existirem
        os.makedirs(os.path.join(output_dir, "left"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "right"), exist_ok=True)
        
        print(f"Carregando SDXL no dispositivo: {device}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
            variant="fp16"
        )
        self.pipe.to(device)
        print("Modelo carregado com sucesso!")

    def generate_pair(self, prompt_a, prompt_b, seed=42, steps=30):
        """
        Gera um par de imagens mantendo a semente (seed) fixa.
        
        No artigo, eles usam Prompt-to-Prompt. Aqui, usei a similaridade estrutural
        fixando o ruído latente inicial (seed=42) para minimizar complexidade e custo computacional.
        """
        # Criar geradores determinísticos para garantir que o "ruído inicial" seja igual
        generator_a = torch.Generator(self.device).manual_seed(seed)
        generator_b = torch.Generator(self.device).manual_seed(seed)
        
        print(f"Gerando Imagem A: '{prompt_a}'")
        image_a = self.pipe(
            prompt=prompt_a, 
            num_inference_steps=steps, 
            generator=generator_a
        ).images[0]

        print(f"Gerando Imagem B: '{prompt_b}'")
        image_b = self.pipe(
            prompt=prompt_b, 
            num_inference_steps=steps, 
            generator=generator_b
        ).images[0]
        
        return image_a, image_b

    def save_pair(self, image_a, image_b, pair_id, prompt_a, prompt_b):
        """Salva as imagens e um metadata simples."""
        path_a = os.path.join(self.output_dir, "left", f"{pair_id}.jpg")
        path_b = os.path.join(self.output_dir, "right", f"{pair_id}.jpg")
        
        image_a.save(path_a)
        image_b.save(path_b)
        
        # Salvar log do par
        log_entry = {
            "id": pair_id,
            "caption_left": prompt_a,
            "caption_right": prompt_b,
            "file_left": path_a,
            "file_right": path_b
        }
        return log_entry

# --- Simulação do "LLM Text Replacement" ---
def simulate_llm_replacement(original_caption):
    """
    Na versão final, aqui entraria a chamada para OpenAI/Llama.
    Para quesito de teste, vou usar um dicionário hardcoded simples.
    """
    replacements = {
        # Animais
        "cat": "dog",
        "horse": "cow",
        "bird": "butterfly",
        
        # Objetos / Casa
        "glass": "bowl",
        "chair": "sofa",
        "laptop": "book",
        
        # Veículos
        "car": "truck",
        "plane": "helicopter",
        "bicycle": "motorcycle",
        
        # Comida / Natureza
        "apple": "orange",
        "tree": "cactus",
        "pizza": "cake"
    }
    
    words = original_caption.split()
    new_words = []
    replaced = False
    
    for word in words:
        clean_word = word.strip(".,")
        if clean_word in replacements and not replaced:
            new_words.append(replacements[clean_word])
            replaced = True
        else:
            new_words.append(word)
            
    return " ".join(new_words)

if __name__ == "__main__":
    generator = ImagePairGenerator()
    
    # Lista de legendas originais (simulando MSCOCO )
    original_captions = [
        "a photo of a cat sitting on a sofa",
        "a car parked on the street",
        "a glass on the table",
        "a plane flying through the air",
        "a brown horse standing in a green field",
        "a bicycle leaning against a brick wall",
        "a large tree standing alone in a desert",
        "a toy plane on a child's bedroom floor"
    ]
    
    metadata_log = []
    
    for i, caption_a in enumerate(original_captions):
        # 1. Gerar a legenda modificada (Simulando LLM)
        caption_b = simulate_llm_replacement(caption_a)
        
        # Se não houve troca, pula
        if caption_a == caption_b:
            print(f"Skipping: '{caption_a}' (nenhuma substituição encontrada)")
            continue
            
        # 2. Gerar o par de imagens
        print(f"\n--- Processando Par {i} ---")
        img_a, img_b = generator.generate_pair(caption_a, caption_b, seed=555+i)
        
        # 3. Salvar
        entry = generator.save_pair(img_a, img_b, f"pair_{i}", caption_a, caption_b)
        metadata_log.append(entry)

    # Salvar metadata final
    with open("data/generated_images/metadata.json", "w") as f:
        json.dump(metadata_log, f, indent=2)
        
    print("\nProcesso finalizado! Imagens salvas em 'data/generated_images'.")