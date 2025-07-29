import json
import re
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import logging
import asyncio
from pathlib import Path

# DSPy imports
import dspy
from dspy import Signature, InputField, OutputField, ChainOfThought, Predict
from attachments.dspy import Attachments

# File processing imports
import pandas as pd
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CategoryMatch:
    """Represents a matched category with its confidence and location"""

    category: str
    confidence: float
    field_name: str
    sample_value: str
    file_source: str
    location: str
    data_type: str
    semantic_meaning: str = ""
    reasoning: str = ""
    retrieval_context: str = ""


@dataclass
class ProcessingResult:
    """Result of processing all files"""

    discovered_categories: Dict[str, List[CategoryMatch]]
    category_mappings: Dict[str, Dict[str, str]]
    context_explanation: str
    confidence_scores: Dict[str, float]
    category_metadata: Dict[str, Dict[str, Any]]
    semantic_analysis: Dict[str, Any]
    category_hierarchy: Dict[str, List[str]]
    agent_message: str
    irena_iterations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IReRaIteration:
    """Represents one iteration of iReRa process"""

    iteration_number: int
    retrieved_patterns: List[str]
    reasoning_output: str
    refined_categories: Dict[str, List[CategoryMatch]]
    confidence_improvement: float
    feedback: str


class IReRaCategorizer:
    """Enhanced categorizer with iReRa implementation"""

    def __init__(
        self,
        lm_model: Optional[dspy.LM] = None,
        max_iterations: int = 3,
        use_lm: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.max_iterations = max_iterations
        self.use_lm = use_lm

        # Initialize DSPy components only if LM is available and enabled
        self.lm_available = False
        if use_lm:
            try:
                if lm_model:
                    dspy.settings.configure(lm=lm_model)
                    self.lm_available = True
                else:
                    self.logger.warning(
                        "No LM model provided. Running in pattern-matching only mode."
                    )
                    self.use_lm = False
            except Exception as e:
                self.logger.error(f"Failed to initialize LM components: {e}")
                self.use_lm = False
                self.lm_available = False

        # Configuration
        self.minimum_confidence = 0.6
        self.convergence_threshold = 0.05
        self.category_similarity_threshold = 0.8

        # Enhanced data type patterns
        self.data_type_patterns = {
            "person_name": [
                r"^[A-Z][a-z]+\s+[A-Z][a-z]+$",
                r"^[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+$",
                r"^(Dr|Mr|Ms|Mrs)\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+$",
            ],
            "date": [
                r"\d{4}-\d{2}-\d{2}",
                r"\d{2}/\d{2}/\d{4}",
                r"\d{2}-\d{2}-\d{4}",
                r"\d{4}/\d{2}/\d{2}",
                r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}",
                r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
            ],
            "phone": [
                r"\(\d{3}\)\s*\d{3}-\d{4}",
                r"\d{3}-\d{3}-\d{4}",
                r"\d{10}",
                r"\+\d{1,3}\s*\d{3,4}\s*\d{3,4}\s*\d{4}",
            ],
            "email": [r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"],
            "address": [
                r"\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd)",
                r"\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd)",
            ],
            "currency": [
                r"^\$?\d+\.?\d*$",
                r"^\d{1,3}(,\d{3})*(\.\d{2})?$",
                r"^\$\d{1,3}(,\d{3})*(\.\d{2})?$",
            ],
            "percentage": [
                r"^\d+\.?\d*%$",
                r"^\d+\.\d+$",
            ],
            "id_number": [
                r"^[A-Z0-9]{6,12}$",
                r"^\d{8,12}$",
                r"^[A-Z]{2,3}\d{6,10}$",
            ],
            "boolean": [r"^(true|false|yes|no|y|n|0|1)$"],
            "numeric": [
                r"^\d+\.?\d*$",
            ],
            "text": [],
        }

    async def categorize_with_irena(
        self, attachments: Attachments
    ) -> ProcessingResult:
        """Main method implementing iReRa for document categorization"""
        self.logger.info(
            f"Starting categorization with attachments (LM enabled: {self.use_lm})"
        )

        # Step 1: Process attachments and extract initial data
        files_data = await self._process_attachments_async(attachments)
        self.logger.info(f"Processed {len(files_data)} files")

        # Step 2: Initialize categories with basic pattern matching
        initial_categories = self._extract_initial_categories(files_data)
        self.logger.info(f"Found {len(initial_categories)} initial categories")

        # Step 3: Generate final results
        final_result = await self._generate_final_results(
            initial_categories, files_data, []
        )

        return final_result

    async def _process_attachments_async(
        self, attachments: Attachments
    ) -> List[Dict[str, Any]]:
        """Process attachments using DSPy Attachments API"""
        files_data = []

        # Always try manual processing first to ensure we get some data
        files_data = await self._process_attachments_manual(attachments)

        print(f"\n📊 FINAL ATTACHMENT PROCESSING RESULTS:")
        print(f"   📁 Total files processed: {len(files_data)}")
        for i, file_data in enumerate(files_data):
            print(f"   📄 File {i}: {file_data.get('file_id', 'unknown')}")
            print(f"      📋 Total fields: {len(file_data)}")
            for key, value in file_data.items():
                if key not in [
                    "file_id",
                    "file_type",
                    "raw_content",
                    "error",
                    "processing_error",
                ]:
                    print(f"         {key}: {str(value)[:50]}...")
                elif key in ["error", "processing_error"]:
                    print(f"         ❌ {key}: {value}")

        return files_data

    async def _process_attachments_manual(
        self, attachments: Attachments
    ) -> List[Dict[str, Any]]:
        """Fallback manual processing of attachments"""
        files_data = []

        print(f"🔍 Manual processing - attachment type: {type(attachments)}")
        print(
            f"🔍 Available attributes: {[attr for attr in dir(attachments) if not attr.startswith('_')]}"
        )

        try:
            # Different ways to access attachments depending on the API
            attachment_list = None

            if hasattr(attachments, "attachments"):
                attachment_list = attachments.attachments
                print(
                    f"✅ Found {len(attachment_list)} attachments via .attachments"
                )
                print(f"   Attachment list contents: {attachment_list}")
            elif hasattr(attachments, "files"):
                attachment_list = attachments.files
                print(f"✅ Found {len(attachment_list)} files via .files")
            elif hasattr(attachments, "__iter__"):
                attachment_list = list(attachments)
                print(f"✅ Found {len(attachment_list)} items via iteration")
            else:
                attachment_list = [attachments]
                print(f"⚠️ Using single attachment object")

            for i, attachment in enumerate(attachment_list):
                try:
                    print(f"\n📄 Processing attachment {i}:")
                    print(f"   Type: {type(attachment)}")
                    print(f"   Value: {attachment}")

                    # If it's a file path (string), read the file directly
                    if isinstance(attachment, str):
                        print(f"   🔄 Reading file path: {attachment}")
                        if os.path.exists(attachment):
                            with open(attachment, "r", encoding="utf-8") as f:
                                content = f.read()
                            filename = os.path.basename(attachment)
                            print(
                                f"   ✅ Read {len(content)} chars from {filename}"
                            )
                        else:
                            print(f"   ❌ File not found: {attachment}")
                            continue
                    else:
                        # Try different ways to get filename and content
                        filename = getattr(
                            attachment, "filename", f"attachment_{i}"
                        )
                        print(f"   Filename: {filename}")

                        # Try different methods to read content
                        content = None
                        if hasattr(attachment, "read"):
                            content = attachment.read()
                            print(
                                f"   ✅ Read content via .read() - {len(str(content))} chars"
                            )
                        elif hasattr(attachment, "content"):
                            content = attachment.content
                            print(
                                f"   ✅ Got content via .content - {len(str(content))} chars"
                            )
                        elif hasattr(attachment, "data"):
                            content = attachment.data
                            print(
                                f"   ✅ Got content via .data - {len(str(content))} chars"
                            )
                        else:
                            content = str(attachment)
                            print(
                                f"   ⚠️ Using str(attachment) - {len(content)} chars"
                            )

                    # Ensure content is string
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="ignore")
                        print(f"   🔄 Decoded bytes to string")

                    print(f"   📝 Content preview: {content[:100]}...")

                    # Determine file type
                    file_type = (
                        Path(filename).suffix.lower().lstrip(".")
                        if "." in filename
                        else "txt"
                    )
                    print(f"   File type: {file_type}")

                    # Process the content
                    processed_data = await self._process_file_content(
                        content, file_type, filename
                    )
                    print(
                        f"   ✅ Processed successfully - found {len(processed_data)} fields"
                    )

                    # Debug: show what fields were found
                    for key, value in processed_data.items():
                        if key not in ["file_id", "file_type", "raw_content"]:
                            print(f"      {key}: {str(value)[:50]}...")

                    files_data.append(processed_data)

                except Exception as e:
                    print(f"   ❌ Error processing attachment {i}: {e}")
                    import traceback

                    traceback.print_exc()
                    files_data.append(
                        {
                            "file_id": f"attachment_{i}",
                            "error": str(e),
                            "content": {},
                        }
                    )

        except Exception as e:
            print(f"❌ Error accessing attachments: {e}")
            import traceback

            traceback.print_exc()
            files_data = [
                {
                    "file_id": "unknown",
                    "error": f"Could not access attachments: {e}",
                    "content": {},
                }
            ]

        print(
            f"\n📊 Manual processing complete: {len(files_data)} files processed"
        )
        return files_data

    async def _process_file_content(
        self, content: str, file_type: str, filename: str
    ) -> Dict[str, Any]:
        """Process individual file content"""
        print(f"\n🔧 Processing file content:")
        print(f"   📄 Filename: {filename}")
        print(f"   📊 Type: {file_type}")
        print(f"   📏 Content length: {len(content)} chars")
        print(f"   📝 Content preview: {content[:200]}...")

        processed_data = {
            "file_id": filename,
            "file_type": file_type,
            "raw_content": content[:1000],
        }

        try:
            if file_type == "json":
                print("   🔄 Processing as JSON...")
                json_data = json.loads(content)
                flattened = self._flatten_dict(json_data)
                processed_data.update(flattened)
                print(f"   ✅ JSON processed - found {len(flattened)} fields")
                for key, value in flattened.items():
                    print(f"      {key}: {str(value)[:50]}...")

            elif file_type == "csv":
                print("   🔄 Processing as CSV...")
                df = pd.read_csv(StringIO(content))
                print(
                    f"   📊 CSV has {len(df.columns)} columns, {len(df)} rows"
                )
                for col in df.columns:
                    sample_value = (
                        df[col].dropna().iloc[0]
                        if not df[col].dropna().empty
                        else ""
                    )
                    processed_data[col.strip()] = str(sample_value)
                    print(f"      {col.strip()}: {str(sample_value)[:50]}...")

            elif file_type in ["txt", "text"]:
                print("   🔄 Processing as text...")
                structured_data = self._extract_structured_data(content)
                processed_data.update(structured_data)
                print(
                    f"   ✅ Text processed - found {len(structured_data)} fields"
                )
                for key, value in structured_data.items():
                    print(f"      {key}: {str(value)[:50]}...")

            else:
                print("   🔄 Processing as unknown type...")
                structured_data = self._extract_structured_data(content)
                processed_data.update(structured_data)
                print(
                    f"   ✅ Unknown type processed - found {len(structured_data)} fields"
                )

        except Exception as e:
            print(f"   ❌ Error processing content: {e}")
            import traceback

            traceback.print_exc()
            processed_data["processing_error"] = str(e)

        print(
            f"   📋 Final processed data has {len(processed_data)} total fields"
        )
        return processed_data

    def _extract_structured_data(self, content: str) -> Dict[str, Any]:
        """Extract structured data from unstructured content"""
        structured_data = {}

        # Look for key-value patterns
        patterns = [
            r"([A-Z][a-zA-Z\s]+?):\s*([^\n\r]+)",
            r"([A-Z][a-zA-Z\s]+?)\s*=\s*([^\n\r]+)",
            r"([A-Z][a-zA-Z\s]+?)\s*[-–]\s*([^\n\r]+)",
            r"([A-Z][a-zA-Z\s]+?)\s*\|\s*([^\n\r]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            for key, value in matches:
                key = key.strip().lower().replace(" ", "_").replace("-", "_")
                value = value.strip()
                if key and value and len(key) < 50 and len(value) < 200:
                    structured_data[key] = value

        # If no structured data found, try to extract common fields
        if not structured_data:
            email_match = re.search(
                r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", content
            )
            if email_match:
                structured_data["email"] = email_match.group(1)

            phone_match = re.search(
                r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", content
            )
            if phone_match:
                structured_data["phone"] = phone_match.group(1)

            date_match = re.search(
                r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", content
            )
            if date_match:
                structured_data["date"] = date_match.group(1)

        return structured_data

    def _extract_initial_categories(
        self, files_data: List[Dict[str, Any]]
    ) -> Dict[str, List[CategoryMatch]]:
        """Extract initial categories using pattern matching"""
        print(
            f"\n🏷️ Extracting initial categories from {len(files_data)} files..."
        )
        categories = defaultdict(list)

        for file_data in files_data:
            file_id = file_data.get("file_id", "unknown")
            print(f"\n📄 Processing file: {file_id}")

            field_count = 0
            for field_name, value in file_data.items():
                if field_name in [
                    "file_id",
                    "file_type",
                    "raw_content",
                    "error",
                    "processing_error",
                ]:
                    continue

                str_value = str(value).strip()
                if not str_value or len(str_value) < 1:
                    print(f"   ⏭️ Skipping empty field: {field_name}")
                    continue

                field_count += 1

                # Detect data type and create category
                data_type = self._detect_data_type(str_value)
                category_name = self._generate_category_name(
                    field_name, data_type
                )

                print(
                    f"   ✅ Field: {field_name} -> Category: {category_name}"
                )
                print(f"      Value: {str_value[:50]}...")
                print(f"      Type: {data_type}")

                match = CategoryMatch(
                    category=category_name,
                    confidence=0.7,
                    field_name=field_name,
                    sample_value=str_value[:100],
                    file_source=file_id,
                    location=field_name,
                    data_type=data_type,
                    reasoning="Initial pattern matching based on field name and data type",
                )

                categories[category_name].append(match)

            print(f"   📊 Found {field_count} fields in {file_id}")

        print(f"\n🎯 Category extraction complete:")
        print(f"   📂 Total categories: {len(categories)}")
        for category_name, matches in categories.items():
            print(f"   📁 {category_name}: {len(matches)} matches")

        return dict(categories)

    def _detect_data_type(self, value: str) -> str:
        """Detect data type of a value"""
        value = value.strip()

        for data_type, patterns in self.data_type_patterns.items():
            for pattern in patterns:
                if re.match(pattern, value, re.IGNORECASE):
                    return data_type
        return "text"

    def _generate_category_name(self, field_name: str, data_type: str) -> str:
        """Generate a category name based on field name and data type"""
        clean_field = field_name.lower().replace("_", " ").replace("-", " ")
        clean_field = re.sub(r"[^a-zA-Z0-9\s]", "", clean_field)
        clean_field = " ".join(clean_field.split())

        if data_type != "text" and data_type not in clean_field:
            return f"{clean_field}_{data_type}"
        return clean_field.replace(" ", "_")

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            self._flatten_dict(
                                item, f"{new_key}[{i}]", sep=sep
                            ).items()
                        )
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        return dict(items)

    async def _generate_final_results(
        self,
        final_categories: Dict[str, List[CategoryMatch]],
        files_data: List[Dict[str, Any]],
        irena_iterations: List[Dict[str, Any]],
    ) -> ProcessingResult:
        """Generate final processing results"""

        # Generate metadata
        category_metadata = self._generate_category_metadata(final_categories)

        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(final_categories)

        # Create file mappings
        file_mappings = self._create_file_mappings(
            files_data, final_categories
        )

        # Generate semantic analysis
        semantic_analysis = self._generate_semantic_analysis(
            final_categories, irena_iterations
        )

        # Generate category hierarchy
        category_hierarchy = self._generate_category_hierarchy(
            final_categories
        )

        # Generate context explanation
        context_explanation = self._generate_context_explanation(
            final_categories, category_metadata, files_data, semantic_analysis
        )

        # Generate message for next agent
        agent_message = self._generate_fallback_agent_message(
            final_categories, confidence_scores, category_metadata
        )

        return ProcessingResult(
            discovered_categories=final_categories,
            category_mappings=file_mappings,
            context_explanation=context_explanation,
            confidence_scores=confidence_scores,
            category_metadata=category_metadata,
            semantic_analysis=semantic_analysis,
            category_hierarchy=category_hierarchy,
            agent_message=agent_message,
            irena_iterations=irena_iterations,
        )

    def _generate_fallback_agent_message(
        self,
        categories: Dict[str, List[CategoryMatch]],
        confidence_scores: Dict[str, float],
        metadata: Dict[str, Dict[str, Any]],
    ) -> str:
        """Generate fallback agent message when LM fails"""
        message = "CATEGORIZATION RESULTS\n"
        message += "=" * 50 + "\n\n"

        message += f"SUMMARY: Discovered {len(categories)} categories across the provided documents.\n\n"

        message += "CATEGORIES DISCOVERED:\n"
        for category_name, matches in categories.items():
            confidence = confidence_scores.get(category_name, 0.0)
            message += f"- {category_name.upper()}: {len(matches)} instances (confidence: {confidence:.2f})\n"

            # Add sample values
            sample_values = list(
                set(match.sample_value for match in matches[:3])
            )
            message += f"  Sample values: {', '.join(sample_values)}\n"

            # Add file distribution
            files = list(set(match.file_source for match in matches))
            message += f"  Found in files: {', '.join(files)}\n\n"

        message += "RECOMMENDATIONS FOR NEXT AGENT:\n"
        message += (
            "1. Use the category mappings to understand document structure\n"
        )
        message += (
            "2. Pay attention to confidence scores when processing data\n"
        )
        message += (
            "3. Consider data type information for validation and processing\n"
        )
        message += "4. Review low-confidence categories for potential manual verification\n"

        return message

    def _generate_category_metadata(
        self, categories: Dict[str, List[CategoryMatch]]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate metadata for each category"""
        metadata = {}
        for category_name, matches in categories.items():
            metadata[category_name] = {
                "total_occurrences": len(matches),
                "unique_files": len(
                    set(match.file_source for match in matches)
                ),
                "average_confidence": sum(
                    match.confidence for match in matches
                )
                / len(matches),
                "data_types": list(set(match.data_type for match in matches)),
                "field_variations": list(
                    set(match.field_name for match in matches)
                ),
                "sample_values": [match.sample_value for match in matches[:3]],
                "confidence_range": {
                    "min": min(match.confidence for match in matches),
                    "max": max(match.confidence for match in matches),
                },
            }
        return metadata

    def _calculate_confidence_scores(
        self, categories: Dict[str, List[CategoryMatch]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for each category"""
        confidence_scores = {}
        for category_name, matches in categories.items():
            if matches:
                confidence_scores[category_name] = sum(
                    match.confidence for match in matches
                ) / len(matches)
            else:
                confidence_scores[category_name] = 0.0
        return confidence_scores

    def _create_file_mappings(
        self,
        files_data: List[Dict[str, Any]],
        categories: Dict[str, List[CategoryMatch]],
    ) -> Dict[str, Dict[str, str]]:
        """Create mappings from files to categories"""
        file_mappings = {}

        for file_data in files_data:
            file_id = file_data.get("file_id", "unknown")
            file_mappings[file_id] = {}

            for category_name, matches in categories.items():
                for match in matches:
                    if match.file_source == file_id:
                        file_mappings[file_id][
                            match.field_name
                        ] = category_name

        return file_mappings

    def _generate_semantic_analysis(
        self,
        categories: Dict[str, List[CategoryMatch]],
        iterations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate semantic analysis of the categorization process"""
        return {
            "total_categories": len(categories),
            "total_iterations": len(iterations),
            "convergence_achieved": len(iterations) < self.max_iterations,
            "category_distribution": {
                name: len(matches) for name, matches in categories.items()
            },
            "average_confidence": self._calculate_overall_confidence(
                categories
            ),
        }

    def _calculate_overall_confidence(
        self, categories: Dict[str, List[CategoryMatch]]
    ) -> float:
        """Calculate overall confidence across all categories"""
        if not categories:
            return 0.0

        total_confidence = 0.0
        total_matches = 0

        for matches in categories.values():
            for match in matches:
                total_confidence += match.confidence
                total_matches += 1

        return total_confidence / total_matches if total_matches > 0 else 0.0

    def _generate_category_hierarchy(
        self, categories: Dict[str, List[CategoryMatch]]
    ) -> Dict[str, List[str]]:
        """Generate category hierarchy"""
        hierarchy = {}

        for category_name, matches in categories.items():
            data_types = [match.data_type for match in matches]
            main_type = max(set(data_types), key=data_types.count)

            if main_type not in hierarchy:
                hierarchy[main_type] = []
            hierarchy[main_type].append(category_name)

        return hierarchy

    def _generate_context_explanation(
        self,
        categories: Dict[str, List[CategoryMatch]],
        metadata: Dict[str, Dict[str, Any]],
        files_data: List[Dict[str, Any]],
        semantic_analysis: Dict[str, Any],
    ) -> str:
        """Generate comprehensive context explanation"""
        explanation = f"DOCUMENT CATEGORIZATION ANALYSIS\n"
        explanation += f"Processed {len(files_data)} files and discovered {len(categories)} distinct categories.\n\n"

        explanation += "CATEGORY SUMMARY:\n"
        for category_name, meta in metadata.items():
            explanation += f"• {category_name}: {meta['total_occurrences']} occurrences across {meta['unique_files']} files "
            explanation += (
                f"(avg confidence: {meta['average_confidence']:.2f})\n"
            )

        return explanation


def create_attachments_from_content(file_data: Dict[str, str]) -> Attachments:
    """Create Attachments object from content dictionary"""
    print(f"🔧 Creating attachments from {len(file_data)} files:")
    for filename, content in file_data.items():
        print(f"   📄 {filename}: {len(content)} characters")

    import tempfile

    temp_files = []
    temp_dir = None

    try:
        temp_dir = tempfile.mkdtemp()
        print(f"   📁 Created temp directory: {temp_dir}")

        # Create temporary files
        for filename, content in file_data.items():
            temp_path = os.path.join(temp_dir, filename)
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
            temp_files.append(temp_path)
            print(f"   ✅ Created temp file: {temp_path}")

        # Try to create Attachments
        try:
            attachments = Attachments()
            print(f"   📋 Basic Attachments created: {type(attachments)}")

            if hasattr(attachments, "attachments") and hasattr(
                attachments.attachments, "append"
            ):
                for temp_path in temp_files:
                    attachments.attachments.append(temp_path)
                    print(f"   ✅ Added {temp_path} via .attachments.append()")

            return attachments

        except Exception as e:
            print(f"   ❌ Attachments creation failed: {e}")

    except Exception as e:
        print(f"   ❌ Temp file creation failed: {e}")

    # Fallback: create a mock attachments object
    print("   🆘 Creating mock attachments object...")

    class MockAttachment:
        def __init__(self, filename, content):
            self.filename = filename
            self.content = content

        def read(self):
            return self.content

        def __str__(self):
            return f"MockAttachment({self.filename})"

    class MockAttachments:
        def __init__(self, file_data):
            self.attachments = [
                MockAttachment(fname, content)
                for fname, content in file_data.items()
            ]
            self.files = self.attachments
            self.metadata = {}
            self.images = []
            self.text = ""

        def __iter__(self):
            return iter(self.attachments)

        def __len__(self):
            return len(self.attachments)

    mock_attachments = MockAttachments(file_data)
    print(f"   ✅ Created mock attachments with {len(mock_attachments)} files")

    # Clean up temp files
    try:
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        if temp_dir and os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass

    return mock_attachments


def setup_categorizer_pattern_only() -> IReRaCategorizer:
    """Setup categorizer without LM (pattern matching only)"""
    return IReRaCategorizer(use_lm=False, max_iterations=1)


def setup_categorizer_with_openai(api_key: str) -> IReRaCategorizer:
    """Setup categorizer with OpenAI"""
    try:
        lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=api_key)
        return IReRaCategorizer(lm_model=lm, max_iterations=3, use_lm=True)
    except Exception as e:
        print(f"Failed to setup OpenAI: {e}")
        return IReRaCategorizer(use_lm=False)


async def categorize_user_files(
    file_paths: List[str], use_openai: bool = False, openai_key: str = ""
) -> ProcessingResult:
    """Categorize user-provided files"""
    print(f"\n🚀 CATEGORIZING USER FILES")
    print("=" * 50)

    # Setup categorizer
    if use_openai and openai_key:
        print("🧠 Using OpenAI-powered categorization...")
        categorizer = setup_categorizer_with_openai(openai_key)
    else:
        print("🎯 Using pattern-based categorization...")
        categorizer = setup_categorizer_pattern_only()

    # Read and process files
    file_data = {}
    valid_files = []

    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue

            print(f"📄 Reading: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            filename = os.path.basename(file_path)
            file_data[filename] = content
            valid_files.append(file_path)
            print(f"   ✅ Read {len(content)} characters")

        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            continue

    if not file_data:
        print("❌ No valid files found!")
        return None

    print(f"\n📊 Processing {len(file_data)} files...")

    # Create attachments and process
    attachments = create_attachments_from_content(file_data)
    result = await categorizer.categorize_with_irena(attachments)

    return result


def get_file_paths_from_user() -> List[str]:
    """Interactive file path input from user"""
    print("\n📁 FILE INPUT OPTIONS:")
    print("=" * 30)
    print("1. Enter file paths manually")
    print("2. Enter directory path (process all files)")
    print("3. Use current directory")

    choice = input("\nChoose option (1-3): ").strip()

    if choice == "1":
        # Manual file paths
        file_paths = []
        print("\nEnter file paths (one per line, empty line to finish):")
        while True:
            path = input("File path: ").strip()
            if not path:
                break
            file_paths.append(path)
        return file_paths

    elif choice == "2":
        # Directory path
        dir_path = input("\nEnter directory path: ").strip()
        if not os.path.exists(dir_path):
            print(f"❌ Directory not found: {dir_path}")
            return []

        file_paths = []
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                file_paths.append(file_path)

        print(f"Found {len(file_paths)} files in directory")
        return file_paths

    elif choice == "3":
        # Current directory
        current_dir = os.getcwd()
        file_paths = []
        for file_name in os.listdir(current_dir):
            if os.path.isfile(file_name) and not file_name.startswith("."):
                file_paths.append(file_name)

        print(f"Found {len(file_paths)} files in current directory")
        return file_paths

    else:
        print("Invalid choice!")
        return []


def display_categorization_results(result: ProcessingResult):
    """Display categorization results in a user-friendly format"""
    if not result:
        print("❌ No results to display")
        return

    print("\n" + "=" * 60)
    print("🎯 CATEGORIZATION RESULTS")
    print("=" * 60)

    # Summary
    total_categories = len(result.discovered_categories)
    total_matches = sum(
        len(matches) for matches in result.discovered_categories.values()
    )
    total_files = len(result.category_mappings)

    print(f"\n📊 SUMMARY:")
    print(f"   📂 Total Categories: {total_categories}")
    print(f"   🔍 Total Matches: {total_matches}")
    print(f"   📄 Files Processed: {total_files}")

    if not result.discovered_categories:
        print("\n⚠️ No categories found in the provided files.")
        print(
            "   Try files with more structured data (JSON, CSV, or text with key-value pairs)"
        )
        return

    # Categories
    print(f"\n📁 DISCOVERED CATEGORIES:")
    print("-" * 40)
    for category, matches in result.discovered_categories.items():
        confidence = result.confidence_scores.get(category, 0.0)
        status = (
            "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        )

        print(f"\n{status} {category.upper()} (confidence: {confidence:.2f})")

        # Show unique sample values
        unique_values = list(set(match.sample_value for match in matches))[:3]
        print(f"   Sample values: {', '.join(unique_values)}")

        # Show files containing this category
        files = list(set(match.file_source for match in matches))
        print(f"   Found in: {', '.join(files)}")

        # Show data type
        data_types = list(set(match.data_type for match in matches))
        print(f"   Data type: {', '.join(data_types)}")

    # File mappings
    print(f"\n📄 FILE BREAKDOWN:")
    print("-" * 40)
    for file_id, mappings in result.category_mappings.items():
        if mappings:
            print(f"\n📋 {file_id}:")
            for field, category in mappings.items():
                print(f"   {field} → {category}")
        else:
            print(f"\n📋 {file_id}: No fields categorized")

    # Agent message
    print(f"\n🤖 AGENT SUMMARY:")
    print("-" * 40)
    print(result.agent_message)


async def interactive_categorization():
    """Interactive categorization workflow"""
    print("🚀 INTERACTIVE FILE CATEGORIZATION SYSTEM")
    print("=" * 50)

    # Get files from user
    file_paths = get_file_paths_from_user()

    if not file_paths:
        print("❌ No files selected!")
        return

    # Check if user wants to use OpenAI
    use_ai = input("\nUse AI-powered categorization? (y/N): ").strip().lower()
    openai_key = ""

    if use_ai in ["y", "yes"]:
        openai_key = input("Enter OpenAI API key: ").strip()
        if not openai_key:
            print("⚠️ No API key provided, using pattern-based categorization")
            use_ai = False
        else:
            use_ai = True
    else:
        use_ai = False

    # Process files
    try:
        result = await categorize_user_files(file_paths, use_ai, openai_key)

        # Display results
        display_categorization_results(result)

        # Ask if user wants to save results
        save_results = (
            input("\n💾 Save results to file? (y/N): ").strip().lower()
        )
        if save_results in ["y", "yes"]:
            save_categorization_results(result, file_paths)

    except Exception as e:
        print(f"❌ Error during categorization: {e}")
        import traceback

        traceback.print_exc()


def save_categorization_results(
    result: ProcessingResult, file_paths: List[str]
):
    """Save categorization results to a JSON file"""
    try:
        # Prepare data for JSON serialization
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "input_files": file_paths,
            "summary": {
                "total_categories": len(result.discovered_categories),
                "total_files": len(result.category_mappings),
                "overall_confidence": result.semantic_analysis.get(
                    "average_confidence", 0.0
                ),
            },
            "categories": {},
            "file_mappings": result.category_mappings,
            "agent_message": result.agent_message,
        }

        # Convert CategoryMatch objects to dictionaries
        for category_name, matches in result.discovered_categories.items():
            output_data["categories"][category_name] = {
                "matches": [
                    {
                        "field_name": match.field_name,
                        "sample_value": match.sample_value,
                        "file_source": match.file_source,
                        "data_type": match.data_type,
                        "confidence": match.confidence,
                        "reasoning": match.reasoning,
                    }
                    for match in matches
                ],
                "metadata": result.category_metadata.get(category_name, {}),
                "confidence_score": result.confidence_scores.get(
                    category_name, 0.0
                ),
            }

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"categorization_results_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Results saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")


async def quick_categorize_command():
    """Quick command-line categorization"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python script.py quick <file1> [file2] [file3] ...")
        print("Example: python script.py quick data.json info.csv report.txt")
        return

    file_paths = sys.argv[2:]  # Skip 'script.py' and 'quick'

    print(f"🚀 QUICK CATEGORIZATION")
    print("=" * 30)
    print(f"Processing {len(file_paths)} files...")

    try:
        result = await categorize_user_files(file_paths, use_openai=False)
        display_categorization_results(result)

    except Exception as e:
        print(f"❌ Error: {e}")


async def batch_process_directory(
    directory_path: str, file_extensions: List[str] = None
):
    """Batch process all files in a directory"""
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return

    if file_extensions is None:
        file_extensions = [".json", ".csv", ".txt", ".xml", ".yaml", ".yml"]

    file_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                file_paths.append(os.path.join(root, file))

    if not file_paths:
        print(
            f"❌ No files found with extensions {file_extensions} in {directory_path}"
        )
        return

    print(f"🔍 Found {len(file_paths)} files to process")

    result = await categorize_user_files(file_paths, use_openai=False)
    display_categorization_results(result)

    # Auto-save for batch processing
    if result:
        save_categorization_results(result, file_paths)


async def process_sample_data() -> ProcessingResult:
    """Process sample data using DSPy Attachments"""
    sample_data = {
        "employee_data.json": json.dumps(
            {
                "employee_id": "EMP001",
                "name": "John Doe",
                "email": "john.doe@company.com",
                "phone": "(555) 123-4567",
                "hire_date": "2023-01-15",
                "salary": "$75000",
                "department": "Engineering",
                "manager": "Jane Smith",
                "address": "123 Main St, Anytown, ST 12345",
            },
            indent=2,
        ),
        "customer_info.csv": """customer_id,name,email,phone,address,registration_date
CUST001,Jane Smith,jane@email.com,555-987-6543,123 Main St,2023-02-01
CUST002,Bob Johnson,bob@test.com,555-111-2222,456 Oak Ave,2023-02-15
CUST003,Alice Williams,alice@company.com,555-333-4444,789 Pine Rd,2023-03-01""",
        "performance_report.txt": """Employee Performance Report
Employee ID: EMP001
Name: Alice Williams
Email: alice@company.com
Performance Score: 85%
Review Date: 2024-03-15
Department: Marketing
Supervisor: Michael Brown
Goals Met: 8/10
Improvement Areas: Time management, Communication
Next Review: 2024-09-15""",
    }

    attachments = create_attachments_from_content(sample_data)
    categorizer = setup_categorizer_pattern_only()
    return await categorizer.categorize_with_irena(attachments)


async def test_categorization_directly():
    """Test categorization directly without attachment complications"""
    print("\n🧪 TESTING CATEGORIZATION DIRECTLY (BYPASSING ATTACHMENTS)")
    print("=" * 60)

    # Create sample data directly
    files_data = [
        {
            "file_id": "employee_data.json",
            "file_type": "json",
            "employee_id": "EMP001",
            "name": "John Doe",
            "email": "john.doe@company.com",
            "phone": "(555) 123-4567",
            "hire_date": "2023-01-15",
            "salary": "$75000",
            "department": "Engineering",
        },
        {
            "file_id": "customer_info.csv",
            "file_type": "csv",
            "customer_id": "CUST001",
            "name": "Jane Smith",
            "email": "jane@email.com",
            "phone": "555-987-6543",
            "address": "123 Main St",
        },
    ]

    print(f"📊 Created {len(files_data)} sample files directly")
    for file_data in files_data:
        print(f"   📄 {file_data['file_id']}: {len(file_data)-2} fields")

    categorizer = setup_categorizer_pattern_only()

    print("\n🏷️ Testing category extraction...")
    categories = categorizer._extract_initial_categories(files_data)

    print(f"\n📊 DIRECT TEST RESULTS:")
    print(f"   Categories found: {len(categories)}")

    if categories:
        for cat_name, matches in categories.items():
            print(f"\n📁 {cat_name.upper()}:")
            for match in matches:
                print(f"   • {match.field_name} = '{match.sample_value}'")
                print(
                    f"     (confidence: {match.confidence:.2f}, type: {match.data_type}, file: {match.file_source})"
                )
    else:
        print("❌ No categories found in direct test!")

    return categories


async def main():
    """Main execution example with better error handling"""
    try:
        print("=" * 60)
        print("STARTING DATA CATEGORIZATION SYSTEM")
        print("=" * 60)

        print("Processing sample data...")
        result = await process_sample_data()

        print("\n" + "=" * 60)
        print("AGENT MESSAGE:")
        print("=" * 60)
        print(result.agent_message)

        print("\n" + "=" * 60)
        print("DISCOVERED CATEGORIES:")
        print("=" * 60)
        if result.discovered_categories:
            for category, matches in result.discovered_categories.items():
                print(f"\n📁 {category.upper()}:")
                for match in matches:
                    print(f"   • {match.field_name} = '{match.sample_value}' ")
                    print(
                        f"     (confidence: {match.confidence:.2f}, type: {match.data_type}, file: {match.file_source})"
                    )
        else:
            print("No categories discovered. Check input data and processing.")

        print("\n" + "=" * 60)
        print("CONFIDENCE SCORES:")
        print("=" * 60)
        for category, score in result.confidence_scores.items():
            status = (
                "🟢 HIGH"
                if score > 0.8
                else "🟡 MEDIUM" if score > 0.6 else "🔴 LOW"
            )
            print(f"  {category}: {score:.2f} {status}")

        print("\n" + "=" * 60)
        print("FILE MAPPINGS:")
        print("=" * 60)
        for file_id, mappings in result.category_mappings.items():
            print(f"\n📄 {file_id}:")
            for field, category in mappings.items():
                print(f"   {field} → {category}")

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)

        return result

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys

    # Verify all classes are defined
    try:
        test_result = ProcessingResult(
            discovered_categories={},
            category_mappings={},
            context_explanation="",
            confidence_scores={},
            category_metadata={},
            semantic_analysis={},
            category_hierarchy={},
            agent_message="",
        )
        print("✅ All classes defined correctly")
    except Exception as e:
        print(f"❌ Class definition error: {e}")
        sys.exit(1)

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "interactive":
            # Interactive mode - user selects files
            asyncio.run(interactive_categorization())

        elif mode == "quick":
            # Quick mode - categorize files from command line
            asyncio.run(quick_categorize_command())

        elif mode == "batch":
            # Batch mode - process entire directory
            if len(sys.argv) > 2:
                directory = sys.argv[2]
                asyncio.run(batch_process_directory(directory))
            else:
                print("Usage: python script.py batch <directory_path>")

        elif mode == "direct":
            # Direct test - bypass attachments
            asyncio.run(test_categorization_directly())

        elif mode == "demo":
            # Demo mode - process sample data
            asyncio.run(main())

        else:
            print("USAGE:")
            print("=" * 50)
            print(
                "python script.py interactive    - Interactive file selection"
            )
            print("python script.py quick <files>  - Quick categorization")
            print("python script.py batch <dir>    - Process entire directory")
            print("python script.py direct         - Test core functionality")
            print(
                "python script.py demo           - Run demo with sample data"
            )
            print("python script.py                - Default (demo mode)")
            print("\nEXAMPLES:")
            print("python script.py interactive")
            print("python script.py quick data.json info.csv")
            print("python script.py batch ./documents")
    else:
        # Default: run interactive mode
        print("🎯 WELCOME TO THE DATA CATEGORIZATION SYSTEM!")
        print("=" * 50)
        print("No arguments provided. Starting interactive mode...")
        print("(Use 'python script.py' followed by a mode for other options)")
        print()
        asyncio.run(interactive_categorization())
