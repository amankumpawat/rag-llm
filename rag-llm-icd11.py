import os
import time
import random
import json
import cProfile
import pstats
import pandas as pd
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, ValidationError
import pdb
import openai

# Debug configuration and structured output
class DebugConfig:
    ENABLED = True
    BREAKPOINTS = False
    COLORS = {
        "error": "\033[91m",
        "warning": "\033[93m",
        "info": "\033[94m",
        "success": "\033[92m",
        "reset": "\033[0m"
    }

class DoctorNote(BaseModel):
    demographics: dict
    doctor_note: str

def validate_json_structure(json_str: str) -> DoctorNote:
    try:
        data = json.loads(json_str)
        return DoctorNote(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        error_msg = f"{DebugConfig.COLORS['error']}JSON Validation Error: {str(e)}{DebugConfig.COLORS['reset']}"
        if DebugConfig.ENABLED:
            error_msg += f"\nInvalid JSON Content:\n{json_str[:500]}..."
        raise ValueError(error_msg)

def print_structured(title: str, data: dict):
    if DebugConfig.ENABLED:
        print(f"\n{DebugConfig.COLORS['info']}=== {title} ===")
        for k, v in data.items():
            print(f"{k.ljust(25)}: {v}")
        print(f"============================={DebugConfig.COLORS['reset']}")

# Original configuration
openai.api_key = "(your_api_key)"

BASE_PATH = '/path/to/folder'
os.makedirs(BASE_PATH, exist_ok=True)

ICD11_PDF_PATH = os.path.join(BASE_PATH, 'ICD-11.pdf')
OUTPUT_CSV_PATH = os.path.join(BASE_PATH, 'ICD-11-Data.csv')
EVALUATION_CSV_PATH = os.path.join(BASE_PATH, 'ICD-11-Evaluation.csv')
PERSIST_DIR = os.path.join(BASE_PATH, 'chroma_db')

llm_chat = ChatOpenAI(
    openai_api_key=openai.api_key,
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Document processing with exclusion of table of contents
loader = PDFPlumberLoader(ICD11_PDF_PATH)
raw_docs = loader.load()[17:517]  # Skip table of contents and take next 500 pages

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
processed_docs = splitter.split_documents(raw_docs)

embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

vector_db = Chroma.from_documents(
    documents=processed_docs,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
    client_settings=chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Dynamic retrieval without fixed k value
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 50}  # High enough to capture all potential mentions
)
qa_chain = RetrievalQA.from_chain_type(llm=llm_chat, retriever=retriever, chain_type="stuff")

# Timing class
class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start
        if DebugConfig.ENABLED:
            print(f"{DebugConfig.COLORS['success']}[TIMING] Completed in {self.duration:.4f} seconds{DebugConfig.COLORS['reset']}")

# Clinical record generation
ICD11_DIAGNOSES = [
    "Depressive episode (6A70)", "Generalized anxiety disorder (6B00)", "Panic disorder (6B01)",
    "Post-traumatic stress disorder (6B40)", "Schizophrenia (6A20)", "Bipolar type I disorder (6A60)",
    "Social anxiety disorder (6B03)", "Obsessive-compulsive disorder (6B20)", "ADHD (6A05)",
    "Anorexia nervosa (6B80)", "Bulimia nervosa (6B81)", "Insomnia disorder (7A00)",
    "Substance dependence (QE50)", "Somatic symptom disorder (6C20)", "Dissociative disorder (6B60)"
]

RACE_OPTIONS = [
    "Caucasian", "African American", "Hispanic", "Asian",
    "Native American", "Middle Eastern", "Pacific Islander", "Mixed"
]
GENDER_OPTIONS = ["Male", "Female", "Non-binary"]

class ClinicalRecordGenerator:
    def __init__(self):
        self.performance_metrics = []

    def _random_diagnosis(self):
        with Timer() as t:
            diagnosis = random.choice(ICD11_DIAGNOSES)
        self.performance_metrics.append(('diagnosis_generation', t.duration))
        return diagnosis

    def _generate_llm_output(self, context, diagnosis, demo):
        prompt = f"""
You are a helpful medical AI. Using the context below from the ICD-11 pdf plus the diagnosis,
write a structured JSON response.

We ALREADY have some demographics for the patient:
- age: {demo["age"]}
- race: {demo["race"]}
- gender: {demo["gender"]}

Please incorporate these EXACT demographics, then provide a
doctor's note summarizing the patient's condition, referencing the diagnosis.

Context from ICD-11:
\"\"\"
{context}
\"\"\"

Diagnosis: {diagnosis}

The final output must be valid JSON with this structure, and nothing else:

{{
  "demographics": {{
    "age": {demo["age"]},
    "race": "{demo["race"]}",
    "gender": "{demo["gender"]}"
  }},
  "doctor_note": "string"
}}
"""
        if DebugConfig.BREAKPOINTS:
            pdb.set_trace()

        try:
            response = llm_chat.predict(prompt)
            validated = validate_json_structure(response)

            if DebugConfig.ENABLED:
                print_structured("LLM Output Validation", {
                    "Status": "Valid",
                    "Diagnosis": diagnosis,
                    "Patient Age": demo["age"],
                    "Response Length": len(response)
                })

            return response
        except Exception as e:
            if DebugConfig.ENABLED:
                print(f"{DebugConfig.COLORS['error']}LLM Error: {str(e)}{DebugConfig.COLORS['reset']}")
            return json.dumps({
                "demographics": demo,
                "doctor_note": f"Error: {str(e)}"
            })

    def generate_record(self, record_id):
        try:
            if DebugConfig.ENABLED:
                print_structured("Record Generation Start", {
                    "Record ID": record_id,
                    "Stage": "Initialization"
                })

            total_timer = Timer()
            total_timer.__enter__()

            diag = self._random_diagnosis()
            base_diagnosis = diag.split('(')[0].strip().lower()

            with Timer() as t:
                # Retrieve all possible matches
                retrieved_docs = qa_chain.retriever.get_relevant_documents(diag)
                
                # Filter for actual mentions of the diagnosis
                filtered_docs = [
                    doc for doc in retrieved_docs
                    if base_diagnosis in doc.page_content.lower()
                ]
                
                # Get actual page numbers from original PDF
                pages = list(set(  # Remove duplicates
                    str(doc.metadata.get("page") + 1) if doc.metadata.get("page") is not None else "Unknown"
                    for doc in filtered_docs
                ))
                
                # Create formatted context with page references
                context_parts = []
                for doc in filtered_docs:
                    page_num = str(doc.metadata.get("page") + 1) if doc.metadata.get("page") is not None else "Unknown"
                    context_parts.append(f"[Page {page_num}]: {doc.page_content[:200]}{'...' if len(doc.page_content) > 200 else ''}")
                formatted_context = "\n\n".join(context_parts)

            self.performance_metrics.append(('rag_retrieval', t.duration))

            demographics = {
                "age": random.randint(18, 70),
                "race": random.choice(RACE_OPTIONS),
                "gender": random.choice(GENDER_OPTIONS)
            }

            with Timer() as t:
                llm_raw_output = self._generate_llm_output(formatted_context, diag, demographics)
                try:
                    llm_data = json.loads(llm_raw_output)
                except json.JSONDecodeError:
                    llm_data = {
                        "demographics": demographics,
                        "doctor_note": "Parsing Error"
                    }
            self.performance_metrics.append(('llm_generation', t.duration))

            rec = {
                "Record ID": record_id,
                "Diagnosis": diag,
                "Age": llm_data["demographics"].get("age"),
                "Race": llm_data["demographics"].get("race"),
                "Gender": llm_data["demographics"].get("gender"),
                "Doctor Note": llm_data["doctor_note"]
            }

            eval_rec = rec.copy()
            eval_rec["Documents Returned"] = len(filtered_docs)
            eval_rec["Context Extracted"] = formatted_context
            eval_rec["Page Numbers"] = pages

            total_timer.__exit__(None, None, None)
            self.performance_metrics.append(('total_generation', total_timer.duration))

            if DebugConfig.ENABLED:
                print_structured("Generated Record", {
                    "Diagnosis": diag,
                    "Context Pages": eval_rec["Page Numbers"],
                    "Mentions Found": len(filtered_docs),
                    "Processing Time": f"{total_timer.duration:.2f}s"
                })

            return rec, eval_rec

        except Exception as e:
            if DebugConfig.ENABLED:
                print(f"{DebugConfig.COLORS['error']}Record Error: {str(e)}{DebugConfig.COLORS['reset']}")
            return {}, {}

def generate_clinical_dataset(patient_count=20):
    generator = ClinicalRecordGenerator()
    dataset, evaluation_data = [], []

    for pid in range(1, patient_count + 1):
        with Timer() as iter_timer:
            row, eval_row = generator.generate_record(pid)
            dataset.append(row)
            evaluation_data.append(eval_row)
            if DebugConfig.ENABLED:
                print(f"{DebugConfig.COLORS['success']}[INFO] Generated record {pid}/{patient_count}{DebugConfig.COLORS['reset']}")

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    eval_df = pd.DataFrame(evaluation_data)
    eval_df.to_csv(EVALUATION_CSV_PATH, index=False)

    return df, eval_df, generator

if __name__ == "__main__":
    # Configure debugging
    DebugConfig.ENABLED = True
    DebugConfig.BREAKPOINTS = False

    # Performance profiling
    profiler = cProfile.Profile()
    profiler.enable()

    try:
        final_dataset, eval_dataset, gen_instance = generate_clinical_dataset(20)

        # Enhanced output formatting
        if DebugConfig.ENABLED:
            print(f"\n{DebugConfig.COLORS['info']}=== Final Dataset Sample ==={DebugConfig.COLORS['reset']}")
            print(final_dataset.head().to_markdown(index=False, tablefmt="grid"))

            print(f"\n{DebugConfig.COLORS['info']}=== Evaluation Metrics ==={DebugConfig.COLORS['reset']}")
            print(eval_dataset.head().to_markdown(index=False, tablefmt="grid"))

            timing_df = pd.DataFrame(gen_instance.performance_metrics,
                                   columns=['Operation', 'Duration'])
            print(f"\n{DebugConfig.COLORS['info']}=== Performance Timings ==={DebugConfig.COLORS['reset']}")
            print(timing_df.groupby('Operation').describe().to_markdown())

    except Exception as e:
        if DebugConfig.ENABLED:
            print(f"{DebugConfig.COLORS['error']}Main Execution Error: {str(e)}{DebugConfig.COLORS['reset']}")
        raise

    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        if DebugConfig.ENABLED:
            print(f"\n{DebugConfig.COLORS['info']}=== Performance Profile ==={DebugConfig.COLORS['reset']}")
            stats.sort_stats('cumtime').print_stats(20)






