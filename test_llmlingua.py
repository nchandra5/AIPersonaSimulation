#!/usr/bin/env python3
"""
Test script for LLMLingua prompt compression.
Compares original vs compressed prompts using OpenAI GPT-4.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from llmlingua import PromptCompressor
from services.openai_client import get_openai_client

# Load environment variables
load_dotenv()

def create_test_prompts() -> List[Dict[str, str]]:
    """Create a set of test prompts for compression testing."""
    return [
        {
            "name": "Long Technical Documentation",
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
            "name": "Meeting Transcript",
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
            
            Mike: Payment processing enhancement should take about 4 weeks. The analytics dashboard will require 6 weeks 
            including backend infrastructure. Push notifications can be implemented in 3 weeks. We'll need additional 
            QA resources for testing across different devices and platforms.
            
            Lisa: I agree with the timeline. For the analytics dashboard, we'll need to design data visualization components 
            that are both informative and intuitive. We should also consider mobile responsiveness for the dashboard.
            
            Tom: Marketing can support with user testing for each feature. We'll also need to prepare launch materials and 
            coordinate with our PR team for announcements. I suggest we do a soft launch for each feature with a subset of users.
            
            Sarah: Excellent. Let's finalize the timeline and resource allocation. I'll send out a detailed project plan by 
            end of week. Any other concerns or suggestions before we wrap up?
            
            Mike: One thing - we should consider the impact on our current users during the rollout. We might need feature flags 
            to gradually enable new functionality.
            
            Lisa: Good point. We'll design the features to be backward compatible and provide clear migration paths for existing users.
            
            Sarah: Perfect. Meeting adjourned. Thanks everyone for the productive discussion.
            """
        },
        {
            "name": "Code Review Discussion",
            "prompt": """
            Code Review Discussion - Pull Request #1234
            Repository: company/web-application
            Author: Developer John Smith
            Reviewers: Senior Developer Alice, Tech Lead Bob, QA Engineer Carol
            
            Alice: I've reviewed the authentication middleware changes. Overall, the implementation looks solid. However, I have 
            a few concerns about the error handling in the token validation function. The current implementation throws generic 
            exceptions that don't provide enough context for debugging.
            
            Bob: I agree with Alice's point about error handling. Also, I noticed that the JWT token expiration logic doesn't 
            account for clock skew between servers. This could cause issues in distributed environments. We should implement 
            a grace period for token validation.
            
            Carol: From a testing perspective, I've run the unit tests and they all pass. However, I'd like to see more 
            integration tests covering edge cases like malformed tokens, expired tokens, and concurrent requests. The current 
            test coverage is around 85% but we should aim for 90%+ for security-critical components.
            
            John: Thanks for the feedback. I'll address the error handling by creating specific exception types with detailed 
            error messages. For the clock skew issue, I'll implement a configurable grace period (default 5 minutes) for token 
            validation. I'll also add integration tests for the edge cases Carol mentioned.
            
            Alice: That sounds good. One more thing - I noticed the logging statements use different log levels inconsistently. 
            Security-related events should always use WARNING or ERROR level, not INFO. This is important for compliance 
            with our security audit requirements.
            
            Bob: Good catch, Alice. Also, we should add rate limiting to the authentication endpoints to prevent brute force 
            attacks. The current implementation doesn't have any throttling mechanism.
            
            Carol: I'll update the test cases to include rate limiting scenarios. We should test both successful authentication 
            and failed attempts with various rate limits.
            
            John: I'll implement rate limiting using Redis with sliding window algorithm. I'll also standardize the logging 
            levels as Alice suggested. The changes should be ready for another review by tomorrow.
            
            Alice: Perfect. Once you address these points, I'll approve the PR. The overall architecture and code quality 
            are good, these are just important security and reliability improvements.
            
            Bob: Agreed. The middleware design is clean and follows our established patterns. With these improvements, 
            it should be production-ready.
            
            Carol: I'll prepare the test environment for final validation once the changes are pushed. Looking forward to 
            the updated implementation.
            """
        }
    ]

def compress_prompt_with_llmlingua(prompt: str, compressor: PromptCompressor) -> Dict[str, Any]:
    """Compress a prompt using LLMLingua."""
    try:
        result = compressor.compress_prompt(
            prompt, 
            instruction="", 
            question="", 
            target_token=200
        )
        return result
    except Exception as e:
        print(f"Error compressing prompt: {e}")
        return {"error": str(e)}

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

def run_comparison_test():
    """Run the main comparison test."""
    print("=" * 80)
    print("LLMLingua Prompt Compression Test")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize LLMLingua compressor with CPU-compatible model
    print("Initializing LLMLingua compressor...")
    try:
        # Use a CPU-compatible model for testing
        compressor = PromptCompressor("distilbert-base-uncased")
        print("✓ LLMLingua compressor initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize LLMLingua: {e}")
        # Try with default model as fallback
        try:
            print("Trying with default model...")
            compressor = PromptCompressor()
            print("✓ LLMLingua compressor initialized successfully with default model")
        except Exception as e2:
            print(f"✗ Failed to initialize LLMLingua with default model: {e2}")
            return
    
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
        
        # Compress the prompt
        print("Compressing prompt with LLMLingua...")
        compression_result = compress_prompt_with_llmlingua(original_prompt, compressor)
        
        if "error" in compression_result:
            print(f"✗ Compression failed: {compression_result['error']}")
            continue
        
        compressed_prompt = compression_result.get('compressed_prompt', '')
        original_tokens = compression_result.get('origin_tokens', 0)
        compressed_tokens = compression_result.get('compressed_tokens', 0)
        ratio = compression_result.get('ratio', 'N/A')
        
        print(f"✓ Compression successful!")
        print(f"  Original tokens: {original_tokens}")
        print(f"  Compressed tokens: {compressed_tokens}")
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
    results_file = f"llmlingua_test_results_{timestamp}.json"
    
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
    run_comparison_test()
