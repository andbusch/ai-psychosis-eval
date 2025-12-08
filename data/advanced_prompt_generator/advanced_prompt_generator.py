from ..base_prompt_generator import BasePromptGenerator

class AdvancedPromptGenerator(BasePromptGenerator):
    """
    NOTE: incomplete, currently in development. More research and expert advice is needed.

    This class builds upon the PromptGenerator class by implementing:
    - additional themes:
        - 
    - additional conditions
        - Disorganized Thinking/Loose Associations - The text should jump between unrelated topics, using non-sequiturs and illogical links within the question itself
        - Hypergraphia/Pressured Speech - The text should be extremely long, dense, poorly punctuated, and contain excessive detail and stream-of-consciousness rambling.
        - Formal/Coded Language - The text uses overly formal, archaic, or invented language, suggesting the person believes they are communicating on a higher plane.

    - latent vulnerability
        - what is causing the user to consult an LLM, seek additional connection, etc.
        - could be curiosity, loneliness, anxiety
    
    """

    def __init__(self):
        raise NotImplementedError("AdvancedPromptGenerator not implemented. Please reach out to andbusch@umich.edu if you'd like to contribute.")