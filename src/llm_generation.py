from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LLMModel:
    def __init__(self):
        model_name = "google/flan-t5-base"
        # Explicitly load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def generate(self, prompt):
        
        # 1. Convert text to numbers (tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 2. Ask the model to generate an answer
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        
        # 3. Convert numbers back into readable text
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)