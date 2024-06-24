class GeneratorSettings:
    def __init__(self, Prompt, NegativePrompt, GuidanceScale, StyleStrength, InferenceSteps):
        self.prompt = Prompt
        self.negative_prompt = NegativePrompt
        self.guidance_scale = GuidanceScale
        self.style_strength = StyleStrength
        self.number_of_steps = InferenceSteps

    def __str__(self):
        return f"Prompt: {self.prompt}\nNegative Prompt: {self.negative_prompt}\nGuidance Scale: {self.guidance_scale}\nStyle Strength: {self.style_strength}\nInference Steps: {self.number_of_steps}"