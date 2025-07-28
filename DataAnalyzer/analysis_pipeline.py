#!/usr/bin/env python3
"""
Unified Document Analysis Pipeline
Combines category discovery and document analysis into a single workflow
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import dspy
from attachments.dspy import Attachments

# Import components from existing modules
from AICategorizerwiReRa.categorizer import (IReRaCategorizer,
                                             ProcessingResult, CategoryMatch,
                                             setup_categorizer_pattern_only,
                                             setup_categorizer_with_openai)
from TrendFinder.modules import trend_analyzer, doc_analyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of the complete pipeline execution"""
    categorization_result: ProcessingResult
    final_csv: str
    final_context: str
    categories_used: List[str]
    documents_processed: int
    pipeline_metadata: Dict[str, Any]


class UnifiedAnalysisPipeline(dspy.Module):
    """
    Unified pipeline that combines category discovery and document analysis
    """

    def __init__(self,
                 lm_model: Optional[dspy.LM] = None,
                 use_openai: bool = False,
                 openai_key: str = "",
                 max_categorization_iterations: int = 3):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        # Initialize categorizer
        if use_openai and openai_key:
            self.logger.info("Initializing with OpenAI-powered categorization")
            self.categorizer = setup_categorizer_with_openai(openai_key)
        elif lm_model:
            self.logger.info("Initializing with provided LM model")
            self.categorizer = IReRaCategorizer(
                lm_model=lm_model,
                max_iterations=max_categorization_iterations,
                use_lm=True)
        else:
            self.logger.info("Initializing with pattern-based categorization")
            self.categorizer = setup_categorizer_pattern_only()

        # Initialize document analyzer
        self.trend_analyzer = trend_analyzer()

        # Pipeline configuration
        self.include_confidence_scores = True
        self.minimum_confidence_threshold = 0.6

    async def forward(
            self,
            documents: List[Attachments],
            initial_context: str = "",
            predefined_categories: Optional[List[str]] = None
    ) -> PipelineResult:
        """
        Main pipeline execution
        
        Args:
            documents: List of document attachments to analyze
            initial_context: Initial context/instructions for the analysis
            predefined_categories: Optional predefined categories to use instead of discovery
            
        Returns:
            PipelineResult containing all outputs and metadata
        """

        self.logger.info(
            f"Starting unified pipeline with {len(documents)} documents")

        # Phase 1: Category Discovery (if not predefined)
        if predefined_categories:
            self.logger.info(
                f"Using predefined categories: {predefined_categories}")
            categories_to_use = predefined_categories
            categorization_result = None
            enhanced_context = initial_context
        else:
            self.logger.info("Phase 1: Discovering categories from documents")

            # Use the first document or all documents for category discovery
            # Your friend's categorizer seems to work with a single Attachments object
            categorization_attachments = documents[
                0] if documents else Attachments()

            categorization_result = await self.categorizer.categorize_with_irena(
                categorization_attachments)

            # Extract categories from categorization result
            categories_to_use = list(
                categorization_result.discovered_categories.keys())

            # Enhance context with categorization insights
            enhanced_context = self._build_enhanced_context(
                initial_context, categorization_result)

            self.logger.info(
                f"Discovered {len(categories_to_use)} categories: {categories_to_use}"
            )

        # Filter categories by confidence if we have categorization results
        if categorization_result and self.include_confidence_scores:
            filtered_categories = self._filter_categories_by_confidence(
                categories_to_use, categorization_result)
            self.logger.info(
                f"Filtered to {len(filtered_categories)} high-confidence categories"
            )
            categories_to_use = filtered_categories

        # Phase 2: Document Analysis and Data Extraction
        self.logger.info("Phase 2: Analyzing documents and extracting data")

        if not categories_to_use:
            self.logger.warning(
                "No categories available for document analysis")
            return PipelineResult(
                categorization_result=categorization_result,
                final_csv="",
                final_context=enhanced_context,
                categories_used=[],
                documents_processed=0,
                pipeline_metadata={"error": "No categories found or provided"})

        # Run trend analysis on all documents
        final_csv, final_context = self.trend_analyzer.forward(
            documents=documents,
            categories=categories_to_use,
            context=enhanced_context)

        # Generate pipeline metadata
        pipeline_metadata = self._generate_pipeline_metadata(
            categorization_result, categories_to_use, documents, final_csv)

        self.logger.info("Pipeline execution completed successfully")

        return PipelineResult(categorization_result=categorization_result,
                              final_csv=final_csv,
                              final_context=final_context,
                              categories_used=categories_to_use,
                              documents_processed=len(documents),
                              pipeline_metadata=pipeline_metadata)

    def _build_enhanced_context(
            self, initial_context: str,
            categorization_result: ProcessingResult) -> str:
        """Build enhanced context from categorization results"""

        enhanced_context = initial_context

        if initial_context:
            enhanced_context += "\n\n"

        enhanced_context += "=== CATEGORY DISCOVERY INSIGHTS ===\n"
        enhanced_context += f"Discovered {len(categorization_result.discovered_categories)} categories from the document collection.\n\n"

        # Add high-confidence categories
        high_conf_categories = [
            cat
            for cat, score in categorization_result.confidence_scores.items()
            if score > 0.8
        ]
        if high_conf_categories:
            enhanced_context += f"High-confidence categories: {', '.join(high_conf_categories)}\n"

        # Add category metadata insights
        enhanced_context += "\nCategory Details:\n"
        for category, metadata in categorization_result.category_metadata.items(
        ):
            enhanced_context += f"- {category}: {metadata['total_occurrences']} occurrences across {metadata['unique_files']} files\n"

        # Add any specific insights from the categorization agent
        if categorization_result.context_explanation:
            enhanced_context += f"\nCategorization Context:\n{categorization_result.context_explanation}\n"

        enhanced_context += "\n=== DOCUMENT ANALYSIS PHASE ===\n"
        enhanced_context += "Now analyzing each document to extract data for the discovered categories.\n"

        return enhanced_context

    def _filter_categories_by_confidence(
            self, categories: List[str],
            categorization_result: ProcessingResult) -> List[str]:
        """Filter categories based on confidence threshold"""

        filtered = []
        for category in categories:
            confidence = categorization_result.confidence_scores.get(
                category, 0.0)
            if confidence >= self.minimum_confidence_threshold:
                filtered.append(category)
            else:
                self.logger.info(
                    f"Filtering out low-confidence category: {category} (confidence: {confidence:.2f})"
                )

        return filtered

    def _generate_pipeline_metadata(
            self, categorization_result: Optional[ProcessingResult],
            categories_used: List[str], documents: List[Attachments],
            final_csv: str) -> Dict[str, Any]:
        """Generate metadata about the pipeline execution"""

        metadata = {
            "total_documents":
            len(documents),
            "categories_discovered":
            len(categorization_result.discovered_categories)
            if categorization_result else 0,
            "categories_used":
            len(categories_used),
            "categorization_method":
            "discovery" if categorization_result else "predefined",
            "csv_rows_generated":
            len(final_csv.split('\n')) -
            1 if final_csv else 0,  # -1 for header
        }

        if categorization_result:
            metadata.update({
                "average_category_confidence":
                categorization_result.semantic_analysis.get(
                    "average_confidence", 0.0),
                "categorization_iterations":
                len(categorization_result.irena_iterations),
                "files_analyzed_for_categories":
                len(categorization_result.category_mappings),
            })

        return metadata


class PipelineManager:
    """High-level manager for running the unified pipeline"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def run_pipeline(
            self,
            documents: List[Attachments],
            initial_context: str = "",
            predefined_categories: Optional[List[str]] = None,
            use_openai: bool = False,
            openai_key: str = "",
            lm_model: Optional[dspy.LM] = None) -> PipelineResult:
        """
        Run the complete analysis pipeline
        
        Args:
            documents: Documents to analyze
            initial_context: Initial instructions/context
            predefined_categories: Optional predefined categories
            use_openai: Whether to use OpenAI for categorization
            openai_key: OpenAI API key if using OpenAI
            lm_model: Custom LM model to use
        """

        # Initialize pipeline
        pipeline = UnifiedAnalysisPipeline(lm_model=lm_model,
                                           use_openai=use_openai,
                                           openai_key=openai_key)

        # Run pipeline
        result = await pipeline.forward(
            documents=documents,
            initial_context=initial_context,
            predefined_categories=predefined_categories)

        return result

    def display_results(self, result: PipelineResult):
        """Display pipeline results in a user-friendly format"""

        print("\n" + "=" * 60)
        print("🚀 UNIFIED ANALYSIS PIPELINE RESULTS")
        print("=" * 60)

        # Pipeline summary
        metadata = result.pipeline_metadata
        print(f"\n📊 PIPELINE SUMMARY:")
        print(f"   📄 Documents processed: {metadata['total_documents']}")
        print(f"   🏷️  Categories used: {metadata['categories_used']}")
        print(f"   📝 Data rows generated: {metadata['csv_rows_generated']}")
        print(f"   🔍 Method: {metadata['categorization_method']}")

        # Show categories used
        if result.categories_used:
            print(f"\n📂 CATEGORIES ANALYZED:")
            for i, category in enumerate(result.categories_used, 1):
                print(f"   {i}. {category}")

        # Show categorization insights (if available)
        if result.categorization_result:
            print(f"\n🔍 CATEGORIZATION INSIGHTS:")
            print(
                f"   🎯 Categories discovered: {metadata['categories_discovered']}"
            )
            print(
                f"   📈 Average confidence: {metadata.get('average_category_confidence', 0.0):.2f}"
            )
            print(
                f"   🔄 Iterations used: {metadata.get('categorization_iterations', 0)}"
            )

        # Show final CSV (first few rows)
        if result.final_csv:
            print(f"\n📋 GENERATED DATA (first 5 rows):")
            csv_lines = result.final_csv.split('\n')[:6]  # Header + 5 rows
            for line in csv_lines:
                if line.strip():
                    print(f"   {line}")
            if len(result.final_csv.split('\n')) > 6:
                print("   ... ({} more rows)".format(
                    len(result.final_csv.split('\n')) - 6))

        # Show final context (truncated)
        if result.final_context:
            print(f"\n💭 FINAL CONTEXT (truncated):")
            context_preview = result.final_context[:500]
            if len(result.final_context) > 500:
                context_preview += "..."
            print(f"   {context_preview}")

        print("\n" + "=" * 60)

    def save_results(self, result: PipelineResult, output_file: str = None):
        """Save pipeline results to files"""

        if not output_file:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"pipeline_results_{timestamp}"

        # Save CSV
        csv_file = f"{output_file}.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write(result.final_csv)

        # Save full results as JSON
        json_file = f"{output_file}.json"
        results_dict = {
            "final_csv": result.final_csv,
            "final_context": result.final_context,
            "categories_used": result.categories_used,
            "documents_processed": result.documents_processed,
            "pipeline_metadata": result.pipeline_metadata,
        }

        # Add categorization results if available
        if result.categorization_result:
            results_dict["categorization_insights"] = {
                "discovered_categories":
                list(
                    result.categorization_result.discovered_categories.keys()),
                "confidence_scores":
                result.categorization_result.confidence_scores,
                "context_explanation":
                result.categorization_result.context_explanation,
                "agent_message":
                result.categorization_result.agent_message,
            }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ Results saved:")
        print(f"   📊 CSV data: {csv_file}")
        print(f"   📄 Full results: {json_file}")


async def example_usage():
    """Example of how to use the unified pipeline"""

    print("🚀 UNIFIED PIPELINE EXAMPLE")
    print("=" * 50)

    # Create sample documents (you would load real documents here)
    from AICategorizerwiReRa.categorizer import create_attachments_from_content

    sample_data = {
        "employee_data.json":
        json.dumps(
            {
                "employee_id": "EMP001",
                "name": "John Doe",
                "email": "john.doe@company.com",
                "phone": "(555) 123-4567",
                "hire_date": "2023-01-15",
                "salary": "$75000",
                "department": "Engineering",
                "performance_score": "85%"
            },
            indent=2),
        "customer_info.csv":
        """customer_id,name,email,phone,address,registration_date,status
CUST001,Jane Smith,jane@email.com,555-987-6543,123 Main St,2023-02-01,Active
CUST002,Bob Johnson,bob@test.com,555-111-2222,456 Oak Ave,2023-02-15,Inactive""",
        "sales_report.txt":
        """Sales Report Q1 2024
Employee: Alice Williams
Email: alice@company.com
Total Sales: $125000
Commission: $12500
Territory: North Region
Customer Count: 45
Performance Rating: Excellent"""
    }

    # Create attachments
    documents = [create_attachments_from_content(sample_data)]

    # Define initial context
    initial_context = """
    Analyze the provided business documents to extract key information about employees, customers, and sales performance. 
    Focus on identifying patterns in performance, contact information, and business metrics.
    The goal is to create a comprehensive dataset for business intelligence analysis.
    """

    # Initialize pipeline manager
    manager = PipelineManager()

    # Run pipeline
    print("Running pipeline...")
    result = await manager.run_pipeline(
        documents=documents,
        initial_context=initial_context,
        use_openai=
        False  # Set to True and provide key for AI-powered categorization
    )

    # Display results
    manager.display_results(result)

    # Save results
    manager.save_results(result)

    return result


async def run_pipeline_with_files(
        file_paths: List[str],
        initial_context: str = "",
        predefined_categories: Optional[List[str]] = None):
    """
    Run pipeline with actual files
    
    Args:
        file_paths: List of file paths to analyze
        initial_context: Initial context for analysis
        predefined_categories: Optional predefined categories
    """

    # Load files into attachments
    from categorizer import create_attachments_from_content
    import os

    file_data = {}
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            filename = os.path.basename(file_path)
            file_data[filename] = content

    if not file_data:
        print("❌ No valid files found!")
        return None

    documents = [create_attachments_from_content(file_data)]

    # Run pipeline
    manager = PipelineManager()
    result = await manager.run_pipeline(
        documents=documents,
        initial_context=initial_context,
        predefined_categories=predefined_categories)

    manager.display_results(result)
    manager.save_results(result)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run with provided files
        file_paths = sys.argv[1:]
        context = "Analyze the provided documents and extract relevant data into a structured format."

        print(f"📁 Running pipeline with {len(file_paths)} files...")
        asyncio.run(run_pipeline_with_files(file_paths, context))
    else:
        # Run example
        print("📚 Running example with sample data...")
        print(
            "Usage: python anaylsis_pipeline.py <file1> <file2> ... for real files"
        )
        asyncio.run(example_usage())
