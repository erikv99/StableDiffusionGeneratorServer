class GeneratorSettings:
    def __init__(self, Prompt, NegativePrompt, GuidanceScale, StyleStrength, InferenceSteps):
        self.prompt = Prompt
        self.negative_prompt = NegativePrompt
        self.guidance_scale = GuidanceScale
        self.style_strength = StyleStrength
        self.number_of_steps = InferenceSteps
