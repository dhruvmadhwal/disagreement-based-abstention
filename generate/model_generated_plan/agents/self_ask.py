"""
Simplified Self-Ask Agent Implementation

Implements three clear approaches for question answering:
1. Direct answer
2. Full decomposition upfront, then answer each subquestion 
3. Iterative next-step follow-up questions until final answer

No complex regex parsing - just clean LLM interactions.
"""

import re
from importlib import import_module
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from rouge_score import rouge_scorer

from prompts.few_shots import (
    format_model_generated_plan_examples,
    format_planning_examples,
    format_subquestion_examples,
)

@dataclass
class QAStep:
    """Represents a single question-answer step in the Self-Ask chain."""
    question: str
    answer: str
    step_number: int


class SelfAskAgent:
    """
    Simplified Self-Ask agent with three clear approaches.
    """
    
    def __init__(self, model, max_steps: int = 6, temperature: float = 0.2, dataset: str = "generic", model_name: Optional[str] = None, default_usage_meta: Optional[Dict[str, Any]] = None):
        """
        Initialize the Self-Ask agent.
        
        Args:
            model: The language model to use for generation
            max_steps: Maximum number of question-answer steps
            temperature: Temperature for model generation
            dataset: Dataset name for specialized handling
            model_name: Optional model name for special handling (e.g., "Qwen/Qwen3-8B", "Qwen/Qwen3-32B")
        """
        self.model = model
        self.max_steps = max_steps
        self.temperature = temperature
        self.dataset = dataset
        self.model_name = model_name
        self.default_usage_meta = default_usage_meta or {}
        self._create_base_prompt, self._create_direct_prompt = self._load_prompt_creators(dataset)

    def _merge_usage_meta(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = dict(self.default_usage_meta)
        if extra:
            merged.update(extra)
        return merged

    def _load_prompt_creators(self, dataset: str):
        """Load dataset-specific prompt factories with Mintaka fallback."""
        module_name = f"prompts.{dataset}.prompts"
        try:
            module = import_module(module_name)
        except ModuleNotFoundError:
            module = import_module("prompts.mintaka.prompts")
        return module.create_model_generated_base_prompt, module.create_model_generated_direct_prompt
        
    def _is_qwen3_model(self) -> bool:
        """Check if the model is a Qwen3 model that needs /no_think."""
        # Check the explicitly passed model name first
        if self.model_name and self.model_name in ("Qwen/Qwen3-8B", "Qwen/Qwen3-32B"):
            return True
        
        # Fallback to checking the model's attribute
        return (hasattr(self.model, 'model_name') and 
                self.model.model_name in ("Qwen/Qwen3-8B", "Qwen/Qwen3-32B"))
    
    def _add_no_think_if_qwen3(self, prompt: str) -> str:
        """Add /no_think to prompt if using Qwen3 model."""
        if self._is_qwen3_model():
            if prompt.lstrip().startswith("/no_think"):
                return prompt
            return f"/no_think\n\n{prompt}"
        return prompt
    
    def _clean_model_output(self, text: str) -> str:
        """Clean model output by removing thinking tags and other artifacts."""
        import re
        
        if not text:
            return ""
        
        # Remove all thinking tag variations and content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)  # <think>...</think>
        text = re.sub(r'<think/>', '', text, flags=re.DOTALL)  # <think/>
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)  # <think> (unclosed, remove rest)
        text = re.sub(r'</think>', '', text, flags=re.DOTALL)  # </think> (orphaned)
        
        # Extract content after "Answer:" if it exists
        answer_match = re.search(r'Answer:\s*(.+)', text, flags=re.IGNORECASE|re.DOTALL)
        if answer_match:
            text = answer_match.group(1)
        
        # Basic whitespace cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _is_response_informative(self, response: str, question: str) -> bool:
        """Check if a response is informative and useful for consistency evaluation."""
        if not response:
            return False
        
        response_lower = response.lower().strip()
        
        # Check for uncertainty patterns
        uncertainty_patterns = [
            r'(?i)(?:I\s*(?:am\s*)?(?:don\'t|do\s*not|cannot|can\'t)\s*(?:know|have|provide|determine))',
            r'(?i)(?:I\'m\s*(?:not\s*)?(?:sure|aware|able|unable))',
            r'(?i)(?:I\s*am\s*(?:not\s*)?(?:sure|aware|able|unable))',
            r'(?i)(?:insufficient|not\s*enough)\s*(?:information|data|context)',
            r'(?i)(?:real-time\s*data\s*access|access\s*to\s*(?:real-time|current)\s*data)',
            r'(?i)(?:unable\s*to\s*(?:provide|determine|identify))',
            r'(?i)(?:unclear|ambiguous|difficult\s*to\s*determine)',
            r'(?i)(?:without\s*(?:more|additional|further)\s*(?:context|information|data))',
            r'(?i)(?:as\s*of\s*my\s*(?:last\s*)?(?:update|knowledge))',
            r'(?i)(?:I\s*(?:don\'t|do\s*not)\s*have\s*(?:access\s*to|information\s*about))',
        ]
        
        # Check if response shows uncertainty
        for pattern in uncertainty_patterns:
            if re.search(pattern, response):
                return False
        
        # Check for overly generic responses
        generic_patterns = [
            r'(?i)(?:it\s*depends|varies|depends\s*on)',
            r'(?i)(?:many\s*factors|several\s*factors|various\s*factors)',
            r'(?i)(?:in\s*general|generally\s*speaking|typically)',
            r'(?i)(?:please\s*(?:refer|check|consult|see))',
            r'(?i)(?:refer\s*to\s*(?:official|authoritative)\s*sources)',
        ]
        
        generic_count = 0
        for pattern in generic_patterns:
            if re.search(pattern, response):
                generic_count += 1
        
        # If too many generic patterns, consider uninformative
        if generic_count >= 2:
            return False
        
        return True
    
    def _generate_robust_answer(self, question: str, context: Optional[str] = None, attempt: int = 1, usage_meta: Optional[Dict[str, Any]] = None) -> str:
        """Generate an answer with robustness checks and retry logic."""
        
        # Base prompt creation with dataset-specific formatting
        prompt = self._create_direct_prompt(question, context_paragraphs=context, dataset=self.dataset)
        
        # For subsequent attempts, add instructions to avoid uncertainty
        if attempt > 1:
            prompt += f"""

IMPORTANT: This is attempt #{attempt}. Please provide a specific, factual answer. Do not say you don't know, don't have access to data, or need more information. Based on your training knowledge, give the best answer you can."""
        
        prompt = self._add_no_think_if_qwen3(prompt)
        response = self.model.generate_answer(prompt, temperature=self.temperature, usage_meta=self._merge_usage_meta(usage_meta))
        response = self._clean_model_output(response)
        
        # Check if response is informative
        if not self._is_response_informative(response, question):
            if attempt < 3:  # Max 3 attempts
                print(f"  ⚠️ Attempt {attempt}: Uninformative response, retrying...")
                return self._generate_robust_answer(question, context, attempt + 1)
            else:
                print(f"  ⚠️ After {attempt} attempts, still uninformative. Using best available response.")
        
        return response
        
    def generate_direct_answer(self, question: str, context: Optional[str] = None, usage_meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Approach 1: Generate a direct answer without decomposition.
        Enhanced with robustness checks and retry logic.
        
        Args:
            question: The question to answer
            context: Optional context
            
        Returns:
            Direct answer string
        """
        return self._generate_robust_answer(question, context, usage_meta=usage_meta)
    
    def generate_full_decomposition(self, question: str, context: Optional[str] = None, usage_meta: Optional[Dict[str, Any]] = None) -> Tuple[List[QAStep], str]:
        """
        Approach 2: Generate full question decomposition upfront, then answer each subquestion.
        
        This approach:
        1. Asks the model to predict all subquestions needed
        2. Answers each subquestion in sequence
        3. Aggregates the final answer
        
        Args:
            question: The main question to decompose and answer
            context: Optional context
            
        Returns:
            Tuple of (chain of QA steps, final answer)
        """
        # Step 1: Generate upfront decomposition
        base_prompt = self._create_base_prompt(context_paragraphs=context, dataset=self.dataset)
        planning_examples = format_planning_examples()
        planning_block = ""
        if planning_examples:
            planning_block = f"{planning_examples}\n\n"

        subquestion_examples = format_subquestion_examples()
        qa_example_block = f"\n{subquestion_examples}\n\n" if subquestion_examples else ""

        decomp_prompt = f"""{base_prompt}{planning_block}I need to break down this complex question into simpler subquestions that I can answer step by step.

Question: {question}

List the specific subquestions I need to answer to solve this completely (one per line, numbered). Your response must be in English. Do not say you need to look anything up - just provide the logical breakdown:

1."""
        
        decomp_prompt = self._add_no_think_if_qwen3(decomp_prompt)
        decomposition_response = self.model.generate_answer(decomp_prompt, temperature=self.temperature, usage_meta=self._merge_usage_meta(usage_meta))
        decomposition_response = self._clean_model_output(decomposition_response)
        
        # Extract subquestions (handle both line-by-line and embedded numbering)
        subquestions = []
        
        # First try to split by embedded numbering (e.g., "1. Question 2. Question 3. Question")
        # This handles cases where Qwen3 puts all questions on one line
        text = decomposition_response.strip()
        
        # Split on patterns like " 2.", " 3.", etc. (with space before number)
        parts = re.split(r'\s+(\d+\.)\s*', text)
        
        if len(parts) > 1:
            # We found embedded numbering - process each part
            current_question = parts[0].strip()  # First question (before any "2.", "3.", etc.)
            current_question = re.sub(r'^\d+\.?\s*', '', current_question)  # Remove leading "1."
            if current_question and len(current_question) > 10:
                subquestions.append(current_question)
            
            # Process remaining parts (they come in pairs: number, question)
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    question = parts[i + 1].strip()
                    if question and len(question) > 10:
                        subquestions.append(question)
        else:
            # Fallback to line-by-line parsing
            lines = text.split('\n')
            for line in lines:
                # Remove numbering and clean up
                clean_line = re.sub(r'^\d+\.?\s*', '', line.strip())
                if clean_line and len(clean_line) > 10:
                    subquestions.append(clean_line)
        
        # Store decomposition but don't add it as a reasoning step in the chain
        chain = []

        if not subquestions:
            # Fallback to direct answer if no decomposition
            direct_answer = self.generate_direct_answer(question, context)
            return chain, direct_answer
        
        # Step 2: Answer each subquestion
        accumulated_info = ""
        
        for i, subquestion in enumerate(subquestions[:self.max_steps]):
            # Generate answer for this subquestion with context from previous answers
            # Use the shared system prompt, but add the accumulated info
            sub_prompt_base = self._create_base_prompt(context_paragraphs=context, dataset=self.dataset)

            example_block = qa_example_block
            if accumulated_info:
                full_prompt = f"""Information from previous subquestions:
{accumulated_info}

{sub_prompt_base}{example_block}Question: {subquestion}
Answer:"""
            else:
                full_prompt = f"""{sub_prompt_base}{example_block}Question: {subquestion}
Answer:"""
            
            full_prompt = self._add_no_think_if_qwen3(full_prompt)
            subq_response = self.model.generate_answer(full_prompt, temperature=self.temperature, usage_meta=self._merge_usage_meta(usage_meta))
            subq_answer = self._clean_model_output(subq_response)
            
            # ENHANCEMENT: Check if subquestion answer is informative, retry if not
            if not self._is_response_informative(subq_answer, subquestion):
                print(f"  ⚠️ Subquestion {i+1} gave uninformative answer, retrying...")
                retry_prompt = full_prompt + f"""

IMPORTANT: Please provide a specific, factual answer to this subquestion. Do not say you don't know or need more information. Give the best answer based on your knowledge."""
                
                retry_prompt = self._add_no_think_if_qwen3(retry_prompt)
                subq_response = self.model.generate_answer(retry_prompt, temperature=self.temperature, usage_meta=self._merge_usage_meta(usage_meta))
                subq_answer = self._clean_model_output(subq_response)
                
                if not self._is_response_informative(subq_answer, subquestion):
                    print(f"  ⚠️ Subquestion {i+1} still uninformative after retry")
            
            # Update accumulated info for next subquestion
            accumulated_info += f"- {subquestion} → {subq_answer}\n"
            
            qa_step = QAStep(
                question=subquestion,
                answer=subq_answer,
                step_number=len(chain) + 1
            )
            chain.append(qa_step)
        
        # Step 3: Generate final answer using all subquestion answers
        final_answer = self._aggregate_answers(question, chain, context)
        
        return chain, final_answer
    
    def generate_iterative_chain(self, question: str, context: Optional[str] = None) -> Tuple[List[QAStep], str]:
        """
        Approach 3: Iterative next-step follow-up questions until final answer.
        
        This approach now uses a single prompt to generate a thought (the next subquestion)
        and an action (the answer to that subquestion) in one LLM call per step.
        
        Args:
            question: The main question to answer
            context: Optional context
            
        Returns:
            Tuple of (chain of QA steps, final answer)
        """
        chain = []
        current_info = ""
        seen_questions = set()  # Track questions to prevent repetition
        base_prompt = self._create_base_prompt(context_paragraphs=context, dataset=self.dataset)

        for step in range(self.max_steps):
            # Use model-specific prompting strategies
            model_name = getattr(self.model, 'model_name', '').lower()
            is_llama = 'llama' in model_name
            is_gemma = 'gemma' in model_name
            
            if step == 0:
                if is_llama:
                    # Llama-specific approach: ask for complete decomposition upfront
                    iter_prompt = f"""<s>[INST] Break down this question into 2-4 simple sub-questions and answer each one:

Question: {question}

Format each step as:
Q: [sub-question]
A: [direct answer]

Then conclude with:
The final answer is [complete answer].

Begin: [/INST]"""
                elif is_gemma:
                    # Gemma-specific approach: simpler format with clear expectations
                    iter_prompt = f"""I need to answer this question by breaking it into smaller parts.

Question: {question}

I'll start with the first step:

1. What is my first sub-question?
2. What is the answer to that sub-question?

Step 1:
Question: """
                else:
                    # Qwen/general format (existing)
                    iter_prompt = f"""Question: {question}

I need to break this down step by step. I will ask one subquestion and answer it concisely.

Examples:
Sub-question: Who was the 16th President of the US?
Answer: Abraham Lincoln

Sub-question: What is the highest-grossing film directed by James Cameron?
Answer: Avatar

IMPORTANT: Give only ONE concise answer per subquestion. No explanations.

Sub-question: [Write your first subquestion here]
Answer: [Give only the direct answer here]"""
            else:
                if is_llama:
                    # Llama-specific continuation prompt
                    iter_prompt = f"""<s>[INST] Question: {question}

What I know so far:
{current_info}

Either:
1) If I can now answer the main question completely: "FINAL: [answer]"
2) If I need one more piece of information: "Q: [question]\nA: [answer]"

Choose 1 or 2. Be direct. [/INST]"""
                elif is_gemma:
                    # Gemma-specific continuation prompt: simpler and more direct
                    step_num = len(chain) + 1
                    iter_prompt = f"""Original question: {question}

What I know so far:
{current_info}

Step {step_num}:
Can I answer the original question now? If yes, write "FINAL ANSWER:" followed by the complete answer.
If no, what is my next question and its answer?

Question: """
                else:
                    # Qwen/general format (existing)
                    iter_prompt = f"""Question: {question}

Progress so far:
{current_info}

STRICT FORMAT REQUIRED:

Option 1 - If I can answer the main question:
Final Answer: [complete answer]

Option 2 - If I need more information:
Sub-question: [one clear question]
Answer: [direct answer only]

Choose Option 1 or Option 2. Give NOTHING ELSE."""

            # One LLM call per step
            iter_prompt = self._add_no_think_if_qwen3(iter_prompt)
            response = self.model.generate_answer(iter_prompt, temperature=self.temperature, usage_meta=self._merge_usage_meta(usage_meta))
            response = self._clean_model_output(response)

            # Check for final answer (model-specific patterns)
            final_answer = None
            if is_llama:
                # Llama-specific final answer patterns
                llama_final_patterns = [
                    r"FINAL:\s*(.*?)(?=\n|$)",
                    r"Final Answer:\s*(.*?)(?=\n|$)",
                ]
                for pattern in llama_final_patterns:
                    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                    if match:
                        final_answer = match.group(1).strip()
                        if final_answer and len(final_answer) > 3 and not (final_answer.startswith('[') and final_answer.endswith(']')):
                            return chain, final_answer
                        break
            elif is_gemma:
                # Gemma-specific final answer patterns
                gemma_final_patterns = [
                    r"FINAL ANSWER:\s*(.*?)(?=\n|$)",
                    r"Final answer:\s*(.*?)(?=\n|$)",
                    r"The answer is:\s*(.*?)(?=\n|$)",
                ]
                for pattern in gemma_final_patterns:
                    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                    if match:
                        final_answer = match.group(1).strip()
                        if final_answer and len(final_answer) > 3 and not (final_answer.startswith('[') and final_answer.endswith(']')):
                            return chain, final_answer
                        break
            else:
                # General/Qwen final answer patterns
                general_final_patterns = [
                    r"Final Answer:\s*(.*?)(?=\n\n|\nSub-question|\nAnswer|$)",
                    r"The final answer is:\s*(.*?)(?=\n\n|\nSub-question|\nAnswer|$)",
                    r"Answer:\s*(.*?)(?=\n\n|$)",
                ]
                for pattern in general_final_patterns:
                    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                    if match:
                        final_answer = match.group(1).strip()
                        if final_answer and len(final_answer) > 3 and not (final_answer.startswith('[') and final_answer.endswith(']')):
                            return chain, final_answer
                        break

            # Parse sub-question and answer (model-specific strategies)
            sub_question = None
            sub_answer = None
            
            if is_llama:
                # Clean up Llama's response artifacts first
                clean_response = re.sub(r'\[/INST\]', '', response)
                clean_response = re.sub(r'<s>', '', clean_response)
                clean_response = re.sub(r'</s>', '', clean_response)
                
                # Check for final answer in Llama format first (only if we already have some steps)
                if len(chain) > 0:  # Only look for final answer after processing steps
                    final_patterns = [
                        r"The final answer is[:\s]*(.*?)(?:\[/INST\]|$)",
                        r"Therefore[,\s]*.*?(?:answer is|answer:)\s*(.*?)(?:\[/INST\]|$)",
                    ]
                    for pattern in final_patterns:
                        final_match = re.search(pattern, clean_response, re.IGNORECASE | re.DOTALL)
                        if final_match:
                            final_answer = final_match.group(1).strip().rstrip('.')
                            if final_answer and len(final_answer) > 3:
                                return chain, final_answer
                
                # Parse for next Q:/A: pair that we haven't seen yet
                # Find all Q:/A: pairs and take the next one after what we've already processed
                qa_pairs = []
                lines = clean_response.split('\n')
                current_q = None
                current_a = None
                
                for line in lines:
                    line = line.strip()
                    # Handle "Step N: Q:" format as well as plain "Q:"
                    if line.startswith('Q:') or (line.startswith('Step') and 'Q:' in line):
                        if current_q and current_a:  # Save previous pair
                            qa_pairs.append((current_q, current_a))
                        # Extract question after "Q:"
                        if 'Q:' in line:
                            q_part = line.split('Q:', 1)[1].strip()
                            current_q = q_part
                        else:
                            current_q = line[2:].strip()
                        current_a = None
                    elif line.startswith('A:') and current_q:
                        current_a = line[2:].strip()
                
                # Add the last pair
                if current_q and current_a:
                    qa_pairs.append((current_q, current_a))
                
                # Process all Q/A pairs for Llama (since it generates them all at once)
                if qa_pairs:
                    # If this is the first step and we have multiple pairs, process them all
                    if step == 0 and len(qa_pairs) > 1:
                        for q, a in qa_pairs:
                            if q and a and len(q) >= 5:
                                qa_step = QAStep(
                                    question=q,
                                    answer=a,
                                    step_number=len(chain) + 1
                                )
                                chain.append(qa_step)
                                current_info += f"- {q} → {a}\n"
                        # Skip to final answer processing
                        continue
                    # Take the next unseen Q/A pair for subsequent steps
                    elif len(qa_pairs) > len(chain):
                        sub_question, sub_answer = qa_pairs[len(chain)]
            elif is_gemma:
                # Gemma-specific parsing: simpler pattern matching
                # Look for patterns like "Question: ... Answer: ..." or just text that can be parsed
                lines = response.split('\n')
                current_q = None
                current_a = None
                
                for line in lines:
                    line = line.strip()
                    # Look for question patterns
                    if line.startswith('Question:') or line.startswith('Q:'):
                        if ':' in line:
                            current_q = line.split(':', 1)[1].strip()
                    # Look for answer patterns  
                    elif line.startswith('Answer:') or line.startswith('A:'):
                        if ':' in line:
                            current_a = line.split(':', 1)[1].strip()
                    # Also handle free-form responses where question and answer are on separate lines
                    elif current_q and not current_a and len(line) > 3 and '?' not in line:
                        # This might be the answer to the current question
                        current_a = line
                
                # If we didn't find structured Q/A, try to extract from free-form text
                if not current_q and not current_a:
                    # Look for question-like content
                    question_patterns = [
                        r"(What .*?\?)",
                        r"(Who .*?\?)",
                        r"(Where .*?\?)",
                        r"(When .*?\?)",
                        r"(How .*?\?)",
                        r"(Which .*?\?)",
                        r"(Why .*?\?)",
                    ]
                    
                    for pattern in question_patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            current_q = match.group(1).strip()
                            # Look for text after the question as potential answer
                            remaining_text = response[match.end():].strip()
                            if remaining_text and len(remaining_text) > 2:
                                # Take first sentence/line as answer
                                current_a = remaining_text.split('\n')[0].strip()
                                if len(current_a) > 50:  # Too long, truncate
                                    current_a = current_a[:50] + "..."
                            break
                
                if current_q and current_a:
                    sub_question = current_q
                    sub_answer = current_a
            else:
                # General/Qwen parsing strategies
                # Strategy 1: Standard "Sub-question:/Answer:" format
                sub_question_match = re.search(r"Sub-question:\s*(.*?)(?=\nAnswer)", response, re.IGNORECASE | re.DOTALL)
                answer_match = re.search(r"Answer:\s*(.*?)(?=\n\n|\nSub-question|\nFinal Answer|$)", response, re.IGNORECASE | re.DOTALL)
                
                if sub_question_match and answer_match:
                    sub_question = sub_question_match.group(1).strip()
                    sub_answer = answer_match.group(1).strip()
                
                if not (sub_question and sub_answer):
                    # Strategy 1b: Handle numbered format
                    numbered_match = re.search(r"\d+\.\s*Sub-question:\s*(.*?)(?=\s*Answer:)", response, re.IGNORECASE | re.DOTALL)
                    if numbered_match:
                        sub_question = numbered_match.group(1).strip()
                        answer_match = re.search(r"Answer:\s*(.*?)(?=\n\n|\nSub-question|\nFinal Answer|$)", response, re.IGNORECASE | re.DOTALL)
                        if answer_match:
                            sub_answer = answer_match.group(1).strip()

            if sub_question and sub_answer and len(sub_question) >= 5:
                # Check for repetition to prevent infinite loops
                sub_question_normalized = sub_question.lower().strip()
                if sub_question_normalized in seen_questions:
                    print(f"Warning: Repetitive question detected, breaking loop: {sub_question}")
                    break
                    
                seen_questions.add(sub_question_normalized)
                
                qa_step = QAStep(
                    question=sub_question,
                    answer=sub_answer,
                    step_number=len(chain) + 1
                )
                chain.append(qa_step)
                current_info += f"- {sub_question} → {sub_answer}\n"
            else:
                # If parsing fails, we assume the model is done or confused.
                break
        
        # If the loop finishes without a "Final Answer:", aggregate what we have.
        final_answer = self._aggregate_answers(question, chain, context)
        
        return chain, final_answer
    
    def _aggregate_answers(self, original_question: str, chain: List[QAStep], context: Optional[str] = None, usage_meta: Optional[Dict[str, Any]] = None) -> str:
        """
        Generates a final answer by aggregating the results from a QA chain.
        
        Args:
            original_question: The original question
            chain: List of QA steps
            context: Optional context
            
        Returns:
            Final answer string
        """
        # Consolidate the gathered information
        # All steps in the chain should now be actual reasoning steps (no metadata)
        reasoning_steps = chain
        if not reasoning_steps:
             # If no valid reasoning steps, attempt a direct answer
             return self.generate_direct_answer(original_question, context)

        accumulated_info = "\n".join([f"- {qa.question} → {qa.answer}" for qa in reasoning_steps])
        
        base_prompt = self._create_base_prompt(context_paragraphs=context, dataset=self.dataset)
        few_shot_block = format_model_generated_plan_examples()
        example_text = ""
        if few_shot_block:
            example_text = (
                "Here are examples of how to combine reasoning chains into final answers:\n"
                f"{few_shot_block}\n\n"
            )
        
        # Add format enforcement based on question type
        format_instruction = self._get_format_instruction(original_question)
        
        final_prompt = f"""{base_prompt}{example_text}Based on the following reasoning chain:
        
{accumulated_info}

Now provide a complete and concise answer to the original question: {original_question}
{format_instruction}
Answer:"""
        
        final_prompt = self._add_no_think_if_qwen3(final_prompt)
        final_answer = self.model.generate_answer(final_prompt, temperature=self.temperature, usage_meta=self._merge_usage_meta(usage_meta))
        return self._clean_model_output(final_answer)
    
    def _get_format_instruction(self, question: str) -> str:
        """Generate format instruction based on dataset type."""
        # For coding benchmarks, ensure code output
        if self.dataset in ['humaneval', 'mbpp']:
            return """Your final answer MUST be executable Python code only. Do not provide explanations, comments, or natural language descriptions. Only provide the complete function implementation that can be executed."""
        
        # For FanoutQA, ensure JSON list format for final answer
        if self.dataset.lower() == 'fanoutqa':
            return """

CRITICAL FanoutQA FORMAT REQUIREMENT:
Your response must be ONLY a JSON list of strings, nothing else. Do not wrap it in an object like {"answer": [...]}. Do not include any explanatory text.

Correct format: ["item1", "item2", "item3"]
Wrong format: {"answer": ["item1", "item2", "item3"]}
Wrong format: The answer is ["item1", "item2", "item3"]

Provide ONLY the JSON list:"""
        
        # Default instruction for non-coding questions
        return """Provide a concise, direct answer. Avoid unnecessary explanations, disclaimers, or verbose descriptions. Be factual and to the point."""
    
    def generate_gold_decomposition_answer(self, question: str, gold_subquestions: List[str], context: Optional[str] = None) -> Tuple[List[QAStep], str]:
        """
        Answer using provided gold standard subquestions (for datasets with decompositions).
        
        Args:
            question: Main question
            gold_subquestions: Pre-defined subquestions from dataset
            context: Optional context
            
        Returns:
            Tuple of (chain of QA steps, final answer)
        """
        if not gold_subquestions:
            direct_answer = self.generate_direct_answer(question, context)
            return [], direct_answer
        
        # Answer each gold subquestion
        chain = []
        accumulated_info = ""
        subquestion_examples = format_subquestion_examples()
        qa_example_block = f"\n{subquestion_examples}\n\n" if subquestion_examples else ""
        
        for i, subquestion in enumerate(gold_subquestions):
            # Use the shared prompt creation function for consistency
            base_prompt = self._create_base_prompt(context_paragraphs=context, dataset=self.dataset)
            subq_prompt = f"{base_prompt}{qa_example_block}Question: {subquestion}\nAnswer:"
            
            if accumulated_info:
                full_prompt = f"""Information from previous subquestions:
{accumulated_info}

{subq_prompt}"""
            else:
                full_prompt = subq_prompt
            
            full_prompt = self._add_no_think_if_qwen3(full_prompt)
            subq_response = self.model.generate_answer(full_prompt, usage_meta=self._merge_usage_meta(usage_meta))
            subq_answer = self._clean_model_output(subq_response)
            
            # Update accumulated info for next subquestion
            accumulated_info += f"- {subquestion} → {subq_answer}\n"
            
            qa_step = QAStep(
                question=subquestion,
                answer=subq_answer,
                step_number=i + 1
            )
            chain.append(qa_step)
        
        # Aggregate final answer
        final_answer = self._aggregate_answers(question, chain, context)
        
        return chain, final_answer

    def format_chain_for_output(self, chain: List[QAStep]) -> List[Dict[str, Any]]:
        """
        Format the QA chain for JSON output.
        
        Args:
            chain: List of QA steps
            
        Returns:
            List of dictionaries with question/answer pairs
        """
        # Handle both QAStep objects and dictionaries (for cached results)
        formatted_chain = []
        for step in chain:
            if isinstance(step, QAStep):
                formatted_chain.append({
                    "step": step.step_number,
                    "question": step.question,
                    "answer": step.answer
                })
            elif isinstance(step, dict):
                # Already formatted
                formatted_chain.append(step)
        
        return formatted_chain
    
    def analyze_subquestion_alignment(self, self_ask_questions: List[str], gold_questions: List[str]) -> Dict[str, Any]:
        """
        Analyze alignment between self-generated and gold subquestions.
        
        Args:
            self_ask_questions: Questions generated by Self-Ask approach
            gold_questions: Gold standard subquestions from dataset
            
        Returns:
            Dictionary with alignment metrics
        """
        metrics = {
            "step_count_comparison": {
                "self_ask_steps": len(self_ask_questions),
                "gold_steps": len(gold_questions),
                "step_difference": len(self_ask_questions) - len(gold_questions)
            },
            "question_alignment": {},
            "coverage_analysis": {}
        }
        
        if not self_ask_questions or not gold_questions:
            return metrics
            
        # Question-to-question similarity using ROUGE if available
        if rouge_scorer:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            similarities = []
            
            for self_q in self_ask_questions:
                max_sim = 0
                best_match = ""
                for gold_q in gold_questions:
                    scores = scorer.score(self_q.lower(), gold_q.lower())
                    rouge_l_f1 = scores['rougeL'].fmeasure
                    if rouge_l_f1 > max_sim:
                        max_sim = rouge_l_f1
                        best_match = gold_q
                similarities.append({
                    "self_question": self_q,
                    "best_gold_match": best_match,
                    "similarity_score": max_sim
                })
            
            metrics["question_alignment"] = {
                "individual_alignments": similarities,
                "average_similarity": sum(s["similarity_score"] for s in similarities) / len(similarities),
                "high_alignment_count": sum(1 for s in similarities if s["similarity_score"] > 0.3)
            }
        
        # Coverage analysis - how many gold questions are "covered" by self-ask questions
        if rouge_scorer and gold_questions:
            covered_gold_questions = 0
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            for gold_q in gold_questions:
                max_coverage = 0
                for self_q in self_ask_questions:
                    scores = scorer.score(gold_q.lower(), self_q.lower())
                    max_coverage = max(max_coverage, scores['rougeL'].fmeasure)
                
                if max_coverage > 0.2:  # Threshold for considering a question "covered"
                    covered_gold_questions += 1
            
            metrics["coverage_analysis"] = {
                "covered_gold_questions": covered_gold_questions,
                "total_gold_questions": len(gold_questions),
                "coverage_percentage": covered_gold_questions / len(gold_questions) if gold_questions else 0,
                "coverage_efficiency": covered_gold_questions / len(self_ask_questions) if self_ask_questions else 0
            }
        
        return metrics
