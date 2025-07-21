#!/usr/bin/env python3
"""
Main evaluation script for Prompt Reflexivity Framework
Usage: python run_evaluation.py --model gpt-4 --output results/gpt4_results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from reflexivity_framework import (
    OpenAIModel, AnthropicModel, HuggingFaceModel,
    ReflexivityEvaluator, save_results
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_model(model_name: str, **kwargs):
    """Factory function to create model instances."""
    if model_name.startswith("gpt"):
        return OpenAIModel(model_name, **kwargs)
    elif model_name.startswith("claude"):
        return AnthropicModel(model_name, **kwargs)
    elif "/" in model_name:  # Hugging Face model
        return HuggingFaceModel(model_name, **kwargs)
    else:
        # Default mappings for common model names
        model_mappings = {
            "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
            "gpt-4": "gpt-4",
            "claude-2": "claude-2"
        }
        
        if model_name in model_mappings:
            full_name = model_mappings[model_name]
            if full_name.startswith("gpt"):
                return OpenAIModel(full_name, **kwargs)
            elif full_name.startswith("claude"):
                return AnthropicModel(full_name, **kwargs)
            else:
                return HuggingFaceModel(full_name, **kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Run Reflexivity Evaluation")
    parser.add_argument("--model", required=True, help="Model to evaluate")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--domain", choices=["reasoning", "ethics", "planning", "common_sense"],
                       help="Specific domain to evaluate")
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to evaluate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens per generation")
    parser.add_argument("--device", default="auto", help="Device for local models (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        model_safe_name = args.model.replace("/", "_").replace("-", "_")
        output_path = output_dir / f"reflexivity_{model_safe_name}.json"
    
    try:
        # Create model
        logger.info(f"Initializing model: {args.model}")
        model_kwargs = {"device": args.device} if "/" in args.model else {}
        model = create_model(args.model, **model_kwargs)
        
        # Create evaluator
        evaluator = ReflexivityEvaluator(model)
        
        # Run evaluation
        logger.info(f"Starting evaluation...")
        results = evaluator.evaluate_benchmark(
            domain=args.domain,
            max_prompts=args.max_prompts
        )
        
        if not results:
            logger.error("No results generated!")
            return
        
        # Compute aggregate metrics
        metrics = evaluator.compute_aggregate_metrics(results)
        logger.info("Aggregate Metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Save results
        save_results(results, str(output_path))
        
        # Save metrics separately
        metrics_path = output_path.with_suffix(".metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
