import torch
from datasets import load_dataset
import torch.nn.functional as F
from transformers import VisionTextDualEncoderModel, VisionTextDualEncoderProcessor
import os
from pathlib import Path
from PIL import Image
import json
from datetime import datetime
from clip import CFG

class ImageSearcher:
    def __init__(self, model_path="best_model.pt", output_dir="search_results"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print("Loading dataset...")
        self.ds = load_dataset("Magneto/caption_for_mars_and_rover_image_size_768")
        self.dataset = self.ds['validation']
        
        # Initialize models and processors
        print("Initializing models...")
        self.processor = VisionTextDualEncoderProcessor.from_pretrained(
            CFG.image_backbone,
            CFG.text_backbone
        )
        
        # Load trained CLIP model
        self.model = VisionTextDualEncoderModel.from_pretrained(
            CFG.image_backbone,
            CFG.text_backbone
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Process all images
        print("Processing images...")
        self.image_features = self._preprocess_images()

    def _preprocess_images(self):
        """Process all images and store their features"""
        features = []
        
        for item in self.dataset:
            image = item['image']
            inputs = self.processor(
                images=image, 
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model.get_image_features(
                    inputs['pixel_values'].to(self.device)
                )
                features.append(outputs)
        
        return torch.cat(features, dim=0)

    def search(self, query, top_k=5):
        """Search for images matching the text query"""
        inputs = self.processor(
            text=query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CFG.max_text_tokens_length
        )
        
        with torch.no_grad():
            text_features = self.model.get_text_features(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device)
            )
            
            similarity = F.cosine_similarity(
                text_features.unsqueeze(1),
                self.image_features,
                dim=-1
            )
            
            top_scores, top_indices = similarity[0].topk(top_k)
            
            return [
                (self.dataset[idx], score.item()) 
                for score, idx in zip(top_scores, top_indices)
            ]

    def save_results(self, query, results):
        """Save search results to directory"""
        # Create timestamp for unique folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create sanitized query string for folder name
        query_slug = "".join(x if x.isalnum() else "_" for x in query)[:50]
        search_dir = self.output_dir / f"{timestamp}_{query_slug}"
        search_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "query": query,
            "timestamp": timestamp,
            "results": []
        }

        # Save images and collect metadata
        for idx, (item, score) in enumerate(results, 1):
            # Save image
            image_filename = f"result_{idx}.jpg"
            image_path = search_dir / image_filename
            
            # Convert to PIL Image if necessary and save
            if isinstance(item['image'], Image.Image):
                image = item['image']
            else:
                image = Image.fromarray(item['image'])
            image.save(image_path)

            # Collect metadata for this result
            result_metadata = {
                "filename": image_filename,
                "score": score,
                "short_caption": item['short_caption'],
                "long_caption": item['long_caption'],
                "url": item['url']
            }
            metadata["results"].append(result_metadata)

        # Save metadata to JSON file
        metadata_path = search_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {search_dir}")
        return search_dir

def main():
    # Initialize the searcher
    searcher = ImageSearcher(model_path="best_model.pt", output_dir="search_results")
    
    # Example searches
    queries = [
        "A large crater on Mars surface",
        "Martian rocks and sand",
        "Rover tracks in the Martian soil"
    ]
    
    # Perform searches and save results
    for query in queries:
        print(f"\nSearching for: {query}")
        results = searcher.search(query, top_k=5)
        
        # Save results
        output_dir = searcher.save_results(query, results)
        
        print(f"Top matches saved to: {output_dir}")
        print("Results:")
        for item, score in results:
            print(f"- Score: {score:.3f}")
            print(f"  Caption: {item['short_caption']}\n")

if __name__ == "__main__":
    main()
