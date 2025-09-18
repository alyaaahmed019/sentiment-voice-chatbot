import torch
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

class EmotionDetector:
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def classify(self, text, retrieved_data=None):
        few_shot_prompt = """
You are an expert at emotion classification.
You can classify text into one of: anger, joy, optimism, or sadness.

Examples:
Text: "I can’t believe they ignored me again, this is so frustrating!"
Emotion: anger

Text: "I just got the job, I’m so happy right now!"
Emotion: joy

Text: "Things are tough now, but I know they’ll get better soon."
Emotion: optimism

Text: "I feel so alone, nobody seems to understand me."
Emotion: sadness

Now, classify the new text.
"""
        prompt = few_shot_prompt + f"\nText: {text}\nEmotion:"
        if retrieved_data:
            context = " ".join([d[0] if isinstance(d, tuple) else d for d in retrieved_data])
            prompt = few_shot_prompt + f"\nContext: {context}\nText: {text}\nEmotion:"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.1, do_sample=True)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower().strip()
        emotions = ["anger", "joy", "optimism", "sadness"]
        for emotion in emotions:
            if emotion in response:
                return emotion
        return "neutral"
