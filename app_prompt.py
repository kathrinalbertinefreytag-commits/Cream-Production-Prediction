from app_data import PARAMETER_MAP

def detect_parameter(prompt):
    """detecting keywords of user_prompt"""
    prompt = prompt.lower()
    found = []

    for key, keywords in PARAMETER_MAP.items():
        for word in keywords:
            if word in prompt:
                found.append(key)
                #break #what happens if more than one keyword given?

    return list(set(found))

def build_focus_schema():
    return {
        "type": "object",
        "properties": {
            "focus_analysis": {"type": "string"},
            "improvements": {
                "type": "array",
                "items": {"type": "string"}
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"}
            },
            "references": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["focus_analysis", "improvements", "risks", "references"]
    }

Full_Cream_Schema = {
    "type": "object",
    "properties": {
        "features": {
            "type": "object",
            "properties": {
                "mixing_time": {"type": "string"},
                "temperature": {"type": "string"},
                "stirring_speed": {"type": "string"},
                "fat_content": {"type": "string"},
                "water_content": {"type": "string"},
                "ph_value": {"type": "string"}
            }
        },
        "assessment": {"type": "string"},
        "improvements": {
            "type": "array",
            "items": {"type": "string"}
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"}
        },
        "references": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["features", "assessment", "improvements", "risks", "references"]
}




def build_cream_prompt(features_df, detected):
    return  f"""
                    You are a cosmetic formulation expert.
                    If there is any feature found in here {detected} you MUST only return information about the mentioned features.
                    Analyze the following input parameters for a cream formulation:
                    {features_df}
                    
                    Your task is to return a structured analysis.

                    IMPORTANT RULES:
                    - Return ONLY valid JSON
                    - Do NOT include any text before or after the JSON
                    - Do NOT include explanations outside the JSON
                    - Use EXACTLY the structure below
                    - All keys and strings MUST use double quotes

                    JSON STRUCTURE:
                    {{
                    "features": {{
                        "mixing_time": "<value + short interpretation>",\n
                        "temperature": "<value + short interpretation>",\n
                        "stirring_speed": "<value + short interpretation>",\n
                        "fat_content": "<value + short interpretation>",
                        "water_content": "<value + short interpretation>",
                        "ph_value": "<value + short interpretation>"
                    }},
                    "assessment": "<2-3 sentences explaining the predicted quality in plain English>",
                    "improvements": [
                        "<specific actionable improvement 1>",
                        "<specific actionable improvement 2>",
                        "<specific actionable improvement 3>"
                    ],
                    "risks": [
                            "<specific risk 1>",
                            "<specific risk 2>"
                            ],

                    "references": [
                        "<reference 1>",
                        "<reference 2>",
                        "<reference 3>"
                    ]
                    }}

                    CONTENT REQUIREMENTS:
                    1. FEATURES:
                    - Include each parameter with its given value
                    - Add a short interpretation (e.g. 'slightly high', 'optimal', 'too low')

                    2. ASSESSMENT:
                    - Explain the quality (good / mediocre / bad) in plain English
                    - Mention WHY (link to parameters)

                    3. IMPROVEMENTS:
                    - Give concrete, numeric suggestions where possible
                    - Focus on how to reach "good" quality

                    4. References:
                    - Provide 2–4 realistic, domain-relevant references
                    - These will be displayed separately in the UI under "Sources"

                    FINAL RULE:
                    Return ONLY the JSON object. No markdown, no text, no comments.
                    """



