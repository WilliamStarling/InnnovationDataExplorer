class ImprovedFieldDetector:
    def __init__(self):
        # More specific and contextual patterns
        self.field_patterns = [
            # Address patterns - more specific
            (r"(?:mailing\s+)?address[:\s]*([^\n]+)", "address"),
            (r"(?:zip|postal)\s*code[:\s]*(\d{5}(?:-\d{4})?)", "postal_code"),
            (r"city[:\s]*([A-Za-z\s]+?)(?:,|\s+\d|\n)", "city"),
            (r"state[:\s]*([A-Z]{2}|\w+)", "state"),
            
            # Contact patterns
            (r"(?:phone|tel|telephone)[:\s]*(\(?[\d\s\-\(\)\.]{10,})", "phone"),
            (r"(?:email|e-mail)[:\s]*([^\s]+@[^\s]+)", "email"),
            
            # Name patterns - more specific
            (r"(?:name|contact)[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)", "person_name"),
            (r"organization[:\s]*([A-Z][^:\n]+)", "organization_name"),
            
            # Date patterns
            (r"date[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", "date"),
            
            # Currency - more restrictive
            (r"(?:amount|cost|price|fee)[:\s]*\$?([\d,]+\.?\d{0,2})", "currency"),
        ]
    
    def extract_fields(self, text: str) -> dict[str, str]:
        """Extract fields with context-aware matching"""
        fields = {}
        
        for pattern, field_type in self.field_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match.group(1).strip()
                if value and len(value) > 1:
                    # Create unique field name if multiple of same type
                    field_name = field_type
                    counter = 1
                    while field_name in fields:
                        field_name = f"{field_type}_{counter}"
                        counter += 1
                    
                    fields[field_name] = value
        
        return fields
    
    def validate_field(self, field_type: str, value: str) -> bool:
        """Validate that the extracted value makes sense for the field type"""
        validation_patterns = {
            "postal_code": r"^\d{5}(?:-\d{4})?$",
            "phone": r"[\d\(\)\-\s\.]{10,}",
            "email": r"^[^\s]+@[^\s]+\.[^\s]+$",
            "currency": r"^\d+\.?\d{0,2}$",
            "person_name": r"^[A-Z][a-z]+\s+[A-Z][a-z]+$",
        }
        
        if field_type in validation_patterns:
            return bool(re.match(validation_patterns[field_type], value))
        
        return True  # Default to valid for unknown types

# Example usage showing the difference:
def compare_approaches(pdf_text: str):
    """Show difference between naive and improved approaches"""
    
    # Naive approach (current system)
    naive_result = naive_extract(pdf_text)
    print("NAIVE APPROACH:")
    for field, value in naive_result.items():
        print(f"  {field}: {value}")
    
    # Improved approach
    detector = ImprovedFieldDetector()
    improved_result = detector.extract_fields(pdf_text)
    print("\nIMPROVED APPROACH:")
    for field, value in improved_result.items():
        if detector.validate_field(field, value):
            print(f"  ✅ {field}: {value}")
        else:
            print(f"  ❌ {field}: {value} (validation failed)")

def naive_extract(text: str) -> dict[str, str]:
    """Simulate the current naive approach"""
    # This is what the current system does
    patterns = [r"([A-Z][a-zA-Z\s]+?):\s*([^\n\r]+)"]
    result = {}
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for key, value in matches:
            # Naive data type detection
            if re.match(r"^\d+$", value.strip()):
                data_type = "currency"  # Wrong assumption!
            elif re.match(r"^[A-Z][a-z]+\s+[A-Z][a-z]+", value.strip()):
                data_type = "person_name"
            else:
                data_type = "text"
            
            clean_key = key.lower().replace(" ", "_")
            result[f"{clean_key}_{data_type}"] = value.strip()
    
    return result