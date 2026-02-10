#!/usr/bin/env python3
"""
Simple LLMLingua test with basic text compression simulation.
This demonstrates the concept without requiring heavy model downloads.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from services.openai_client import get_openai_client

# Load environment variables
load_dotenv()

def simple_text_compression(text: str, target_ratio: float = 0.3) -> str:
    """
    Simple text compression simulation for demonstration.
    In a real scenario, LLMLingua would use ML models for intelligent compression.
    """
    # Split into sentences
    sentences = text.split('. ')
    
    # Keep only a portion of sentences (simulating compression)
    keep_count = max(1, int(len(sentences) * target_ratio))
    compressed_sentences = sentences[:keep_count]
    
    # Join back with some truncation
    compressed_text = '. '.join(compressed_sentences)
    
    # Add ellipsis to indicate compression
    if len(compressed_text) < len(text) * 0.8:
        compressed_text += "..."
    
    return compressed_text

def create_test_prompts() -> List[Dict[str, str]]:
    """Create a set of test prompts for compression testing."""
    return [
        {
            "name": "Technical Documentation",
            "prompt": """
            Artificial Intelligence (AI) has revolutionized numerous industries and continues to shape the future of technology. 
            Machine learning algorithms, particularly deep learning neural networks, have enabled breakthroughs in computer vision, 
            natural language processing, and autonomous systems. These systems require massive amounts of data for training, 
            sophisticated computational resources, and careful tuning of hyperparameters to achieve optimal performance.
            
            The development of transformer architectures has been particularly influential, leading to models like GPT, BERT, 
            and their successors that can understand and generate human-like text. These models have applications in chatbots, 
            content generation, translation services, and code assistance tools.
            
            However, AI systems also face challenges including bias in training data, interpretability of decisions, 
            computational costs, and ethical considerations around automation and job displacement. Researchers are actively 
            working on solutions such as federated learning, explainable AI, and responsible AI frameworks.
            
            The future of AI likely involves more efficient models, better integration with human workflows, and continued 
            advances in areas like reinforcement learning, multimodal AI, and artificial general intelligence research.
            """
        },
        {
            "name": "Meeting Summary",
            "prompt": """
            Meeting Transcript - Product Planning Session
            Date: December 15, 2024
            Participants: Sarah (Product Manager), Mike (Engineering Lead), Lisa (Design Lead), Tom (Marketing)
            
            Sarah: Good morning everyone. Thanks for joining today's product planning session. We need to discuss the Q1 roadmap 
            and prioritize features for our mobile app. Mike, can you give us an update on the current development status?
            
            Mike: Sure. We've completed the user authentication system and the core navigation framework. The team is currently 
            working on the payment integration module. We're about 60% complete with that. The main challenge we're facing is 
            integrating with multiple payment providers while maintaining security standards.
            
            Lisa: From a design perspective, we've finalized the wireframes for the main user flows. The design system is 
            consistent across all screens. We're now focusing on micro-interactions and accessibility features. I'd like to 
            propose adding haptic feedback for better user experience.
            
            Tom: Marketing has been conducting user research and we've identified three key user personas. The primary persona 
            is millennials aged 25-35 who value convenience and speed. Our secondary persona is small business owners who need 
            efficient transaction management. We should prioritize features that serve both groups.
            
            Sarah: That's helpful context. Based on your inputs, I propose we focus on three key features for Q1: 
            1) Enhanced payment processing with multiple provider support, 2) Advanced analytics dashboard for business users, 
            and 3) Push notification system for real-time updates. Mike, what's your timeline estimate for these features?
            """
        }
    ]

def call_openai_api(prompt: str, client) -> str:
    """Call OpenAI API with the given prompt."""
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL_CHAT", "gpt-5"),
            reasoning={"effort": os.getenv("OPENAI_REASONING_EFFORT", "low")},
            input=[{"role": "user", "content": prompt}],
        )
        return getattr(response, "output_text", "") or ""
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

def run_simple_test():
    """Run a simple compression test."""
    print("=" * 80)
    print("Simple LLMLingua-style Prompt Compression Test")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    try:
        client = get_openai_client()
        print("✓ OpenAI client initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize OpenAI client: {e}")
        return
    
    print()
    
    # Get test prompts
    test_prompts = create_test_prompts()
    results = []
    
    for i, test_case in enumerate(test_prompts, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print("-" * 60)
        
        original_prompt = test_case['prompt'].strip()
        print(f"Original prompt length: {len(original_prompt)} characters")
        
        # Simulate compression
        print("Simulating prompt compression...")
        compressed_prompt = simple_text_compression(original_prompt, target_ratio=0.4)
        
        original_tokens = len(original_prompt.split())  # Rough token count
        compressed_tokens = len(compressed_prompt.split())
        ratio = f"{original_tokens / compressed_tokens:.1f}x" if compressed_tokens > 0 else "N/A"
        
        print(f"✓ Compression simulated!")
        print(f"  Original tokens (approx): {original_tokens}")
        print(f"  Compressed tokens (approx): {compressed_tokens}")
        print(f"  Compression ratio: {ratio}")
        print()
        
        # Display prompts
        print("ORIGINAL PROMPT:")
        print("-" * 40)
        print(original_prompt[:500] + "..." if len(original_prompt) > 500 else original_prompt)
        print()
        
        print("COMPRESSED PROMPT:")
        print("-" * 40)
        print(compressed_prompt[:500] + "..." if len(compressed_prompt) > 500 else compressed_prompt)
        print()
        
        # Call OpenAI API with both prompts
        print("Calling OpenAI API with original prompt...")
        original_response = call_openai_api(original_prompt, client)
        print("✓ Original prompt response received")
        
        print("Calling OpenAI API with compressed prompt...")
        compressed_response = call_openai_api(compressed_prompt, client)
        print("✓ Compressed prompt response received")
        print()
        
        # Display responses
        print("ORIGINAL PROMPT RESPONSE:")
        print("-" * 40)
        print(original_response[:300] + "..." if len(original_response) > 300 else original_response)
        print()
        
        print("COMPRESSED PROMPT RESPONSE:")
        print("-" * 40)
        print(compressed_response[:300] + "..." if len(compressed_response) > 300 else compressed_response)
        print()
        
        # Store results
        test_result = {
            "test_case": test_case['name'],
            "original_prompt": original_prompt,
            "compressed_prompt": compressed_prompt,
            "compression_stats": {
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "ratio": ratio
            },
            "original_response": original_response,
            "compressed_response": compressed_response,
            "timestamp": datetime.now().isoformat()
        }
        results.append(test_result)
        
        print("=" * 80)
        print()
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simple_compression_test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Results saved to: {results_file}")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total test cases: {len(results)}")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if results:
        avg_compression_ratio = sum(
            float(r['compression_stats']['ratio'].replace('x', '')) 
            for r in results 
            if r['compression_stats']['ratio'] != 'N/A'
        ) / len(results)
        print(f"Average compression ratio: {avg_compression_ratio:.1f}x")
    
    print("=" * 80)

if __name__ == "__main__":
    run_simple_test()

