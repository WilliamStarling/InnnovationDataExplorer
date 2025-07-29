import dspy
from dspy import Signature, InputField, OutputField, ChainOfThought
from typing import List, Dict, Any
import json

class FieldExtractionSignature(Signature):
    """Extract structured fields from document text"""
    document_text: str = InputField(desc="The raw text from a document")
    field_definitions: List[Dict[str, str]] = OutputField(
        desc="List of field definitions with name, type, and sample value"
    )

class CategoryRefinementSignature(Signature):
    """Refine and validate extracted categories"""
    raw_categories: List[Dict[str, str]] = InputField(desc="Initial extracted categories")
    context: str = InputField(desc="Document context and type")
    refined_categories: List[Dict[str, str]] = OutputField(
        desc="Refined categories with corrected names and types"
    )

class ProperDSPyCategorizer:
    def __init__(self, lm_model):
        dspy.settings.configure(lm=lm_model)
        self.field_extractor = ChainOfThought(FieldExtractionSignature)
        self.category_refiner = ChainOfThought(CategoryRefinementSignature)
        
        # Define expected field types for validation
        self.standard_types = {
            "person_name": "Full name of a person",
            "organization_name": "Name of an organization or company", 
            "address": "Street address",
            "city": "City name",
            "state": "State or province",
            "postal_code": "ZIP or postal code",
            "phone": "Phone number",
            "email": "Email address",
            "date": "Date in any format",
            "currency": "Monetary amount",
            "id_number": "Identification or reference number",
            "boolean": "Yes/no or true/false value"
        }
    
    async def categorize_document(self, document_text: str, document_type: str = "unknown") -> Dict[str, Any]:
        """Main categorization method using DSPy"""
        
        # Step 1: Extract fields using LM reasoning
        print("🔍 Extracting fields with language model...")
        
        # Prepare the text for better processing
        clean_text = self._preprocess_text(document_text)
        
        # Use DSPy to extract fields
        extraction_result = self.field_extractor(document_text=clean_text)
        
        # Step 2: Refine categories using context
        print("🔧 Refining categories...")
        
        refinement_result = self.category_refiner(
            raw_categories=extraction_result.field_definitions,
            context=f"Document type: {document_type}. Length: {len(document_text)} chars."
        )
        
        # Step 3: Validate and structure results
        final_categories = self._validate_categories(refinement_result.refined_categories)
        
        return {
            "categories": final_categories,
            "extraction_reasoning": extraction_result.rationale if hasattr(extraction_result, 'rationale') else "",
            "refinement_reasoning": refinement_result.rationale if hasattr(refinement_result, 'rationale') else "",
            "document_type": document_type,
            "confidence_scores": self._calculate_confidence(final_categories)
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and prepare text for better LM processing"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (LM context limits)
        if len(text) > 8000:
            text = text[:8000] + "..."
            
        return text
    
    def _validate_categories(self, raw_categories: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Validate and clean up extracted categories"""
        validated = []
        
        for category in raw_categories:
            # Ensure required fields
            if not isinstance(category, dict) or 'name' not in category:
                continue
                
            name = category.get('name', '').strip()
            field_type = category.get('type', 'text').lower()
            sample_value = category.get('sample_value', '')
            
            # Skip invalid entries
            if not name or len(name) < 2:
                continue
                
            # Normalize field type
            field_type = self._normalize_field_type(field_type)
            
            validated.append({
                'name': name,
                'type': field_type,
                'sample_value': sample_value,
                'description': self.standard_types.get(field_type, f"Field of type {field_type}"),
                'validation_passed': self._validate_field_value(field_type, sample_value)
            })
        
        return validated
    
    def _normalize_field_type(self, field_type: str) -> str:
        """Normalize field type to standard categories"""
        field_type = field_type.lower().strip()
        
        # Map common variations to standard types
        type_mappings = {
            'name': 'person_name',
            'full_name': 'person_name',
            'contact_name': 'person_name',
            'company': 'organization_name',
            'org': 'organization_name',
            'organization': 'organization_name',
            'zip': 'postal_code',
            'zipcode': 'postal_code',
            'zip_code': 'postal_code',
            'telephone': 'phone',
            'tel': 'phone',
            'phone_number': 'phone',
            'mail': 'email',
            'e_mail': 'email',
            'money': 'currency',
            'amount': 'currency',
            'price': 'currency',
            'cost': 'currency',
        }
        
        return type_mappings.get(field_type, field_type)
    
    def _validate_field_value(self, field_type: str, value: str) -> bool:
        """Validate if a value matches its assigned type"""
        import re
        
        if not value:
            return False
            
        validation_patterns = {
            'person_name': r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$',
            'email': r'^[^\s]+@[^\s]+\.[^\s]+$',
            'phone': r'[\d\(\)\-\s\.]{10,}',
            'postal_code': r'^\d{5}(-\d{4})?$',
            'currency': r'^\$?\d+(\.\d{2})?$',
            'date': r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',
        }
        
        if field_type in validation_patterns:
            return bool(re.search(validation_patterns[field_type], value))
        
        return True  # Default to valid for unknown types
    
    def _calculate_confidence(self, categories: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for categories"""
        confidence_scores = {}
        
        for category in categories:
            name = category['name']
            
            # Base confidence
            confidence = 0.7
            
            # Boost for validation
            if category.get('validation_passed', False):
                confidence += 0.2
            
            # Boost for standard types
            if category['type'] in self.standard_types:
                confidence += 0.1
                
            # Cap at 1.0
            confidence = min(confidence, 1.0)
            
            confidence_scores[name] = confidence
        
        return confidence_scores

# Usage example:
async def test_proper_dspy():
    """Test the proper DSPy implementation"""
    
    # Initialize with your LM
    lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key="your-key-here")
    categorizer = ProperDSPyCategorizer(lm)
    
    # Sample PDF-like text
    sample_text = """
    Organization Name: Acme Corporation
    Mailing Address: 123 Business Park Drive
    City: Springfield  
    State: IL
    Postal Code: 62701
    Contact Person: John Smith
    Phone: (555) 123-4567
    Email: jsmith@acme.com
    """
    
    result = await categorizer.categorize_document(sample_text, "business_form")
    
    print("PROPER DSPY RESULTS:")
    for category in result['categories']:
        confidence = result['confidence_scores'].get(category['name'], 0)
        validation = "✅" if category['validation_passed'] else "❌"
        print(f"{validation} {category['name']} ({category['type']}): {category['sample_value']} (confidence: {confidence:.2f})")
