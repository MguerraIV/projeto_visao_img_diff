import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import FastSAM
import clip
import json
import os

class DifferenceDetector:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Carregando modelos no dispositivo: {device}...")
        
        # 1. Carregar FastSAM para segmentação (Citado no paper, seção 3.3.1)
        self.segment_model = FastSAM('FastSAM-x.pt')
        
        # 2. Carregar CLIP para comparação de similaridade (Citado no paper, seção 3.3.2)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        print("Modelos carregados!")

    def get_similarity(self, image_crop_a, image_crop_b):
        """
        Calcula a similaridade de cosseno entre dois recortes de imagem usando CLIP.
        """
        
        img_a = self.clip_preprocess(image_crop_a).unsqueeze(0).to(self.device)
        img_b = self.clip_preprocess(image_crop_b).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_a = self.clip_model.encode_image(img_a)
            feat_b = self.clip_model.encode_image(img_b)
            
            feat_a /= feat_a.norm(dim=-1, keepdim=True)
            feat_b /= feat_b.norm(dim=-1, keepdim=True)
            similarity = (feat_a @ feat_b.T).item()
            
        return similarity

    def detect(self, image_path_a, image_path_b, conf_threshold=0.4, sim_threshold=0.85):
        """
        Pipeline principal de detecção de diferenças.
        sim_threshold=0.85 é o valor sugerido no paper (Seção 15.4) para diferenças.
        """
        # Carregar imagens
        img_a_pil = Image.open(image_path_a).convert("RGB")
        img_b_pil = Image.open(image_path_b).convert("RGB")
        
        # 1. Segmentação na Imagem A para achar candidatos
        # O paper sugere segmentar as duas, mas para substituição de objetos, segmentar a original (A)
        # e checar se mudou na (B) é uma estratégia eficiente.
        results = self.segment_model(image_path_a, device=self.device, conf=conf_threshold, verbose=False)
        
        detected_differences = []
        
        # O resultado do FastSAM pode conter vários objetos
        if len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            print(f"Encontrados {len(boxes)} objetos candidatos. Verificando mudanças...")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # Garantir limites da imagem
                w, h = img_a_pil.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Se a caixa for muito pequena, ignora (ruído)
                if (x2 - x1) < 100 or (y2 - y1) < 100:
                    continue

                # 2. Recortar a mesma região nas duas imagens
                crop_a = img_a_pil.crop((x1, y1, x2, y2))
                crop_b = img_b_pil.crop((x1, y1, x2, y2))
                
                # 3. Comparar visualmente com CLIP
                similarity = self.get_similarity(crop_a, crop_b)
                
                if similarity < sim_threshold:
                    print(f" -> Diferença detectada no obj {i}! Similaridade: {similarity:.3f} (Threshold: {sim_threshold})")
                    detected_differences.append({
                        "bbox": [x1, y1, x2, y2],
                        "similarity": similarity
                    })
        
        # 4. Filtragem NMS (Non-Maximum Suppression) simples
        final_differences = self.filter_overlapping(detected_differences)
        
        return final_differences

    def filter_overlapping(self, differences, iou_thresh=0.5):
        """
        Remove caixas redundantes/sobrepostas.
        """
        if not differences:
            return []
            
        differences.sort(key=lambda x: x['similarity'])
        
        keep = []
        while differences:
            current = differences.pop(0)
            keep.append(current)
            
            def get_iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea)
                return iou

            differences = [d for d in differences if get_iou(current['bbox'], d['bbox']) < iou_thresh]
            
        return keep
    
    def draw_bounding_box(self, image_path, diffs):
        img = cv2.imread(image_path)
        
        x1, y1, x2, y2 = diffs['bbox']
        # Desenha retângulo VERMELHO
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return img
    

    def save_pair(self, image_a, image_b, pair_id, diffs):
        """Salva as imagens"""
        output_dir = "data/caption_images"
        path_a = os.path.join(output_dir, "left", f"caption_{pair_id}.jpg")
        path_b = os.path.join(output_dir, "right", f"caption_{pair_id}.jpg")

        img_a = self.draw_bounding_box(image_a, diffs)
        cv2.imwrite(path_a, img_a)

        img_b = self.draw_bounding_box(image_b, diffs)
        cv2.imwrite(path_b, img_b)

        return path_a.replace("\\", "/"), path_b.replace("\\", "/")
        

if __name__ == "__main__":
    base_dir = "data/generated_images"
    output_dir = "data/raw_captions"
    try:
        samples = os.listdir(os.path.join(base_dir, "left"))
        detector = DifferenceDetector()

        for cont, sample_id in enumerate(samples):
            img_a_path = os.path.join(base_dir, "left", sample_id)
            img_a_path = img_a_path.replace("\\", "/")

            img_b_path = os.path.join(base_dir, "right", sample_id)
            img_b_path = img_b_path.replace("\\", "/")

            diffs = detector.detect(img_a_path, img_b_path)

            print(f"\nResultado Final: {len(diffs)} diferenças encontradas.")
            print(json.dumps(diffs, indent=2))
            
            if diffs:
                best_diff = min(diffs, key=lambda x: x["similarity"])

                json_name = f"pair_{cont}.json"
                json_output = os.path.join(output_dir, json_name)
                json_output = json_output.replace("\\", "/")

                bounded_img_a_path, bounded_img_b_path = detector.save_pair(img_a_path, img_b_path, cont, best_diff)
                best_diff['img_a'] = bounded_img_a_path
                best_diff['img_b'] = bounded_img_b_path
                
                with open(json_output, "w", encoding="utf-8") as f:
                    json.dump(best_diff, f, indent=4, ensure_ascii=False)

                print("Imagens salvas em data/caption_images")
            
    except IndexError:
        print("Erro: Nenhuma imagem encontrada em 'data/generated_images'. Rode o script de geração primeiro.")