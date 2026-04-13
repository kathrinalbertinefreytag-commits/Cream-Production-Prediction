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