import os
import tempfile
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
import streamlit as st
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional, Any, Type
import re
from datetime import datetime
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import io
from dataclasses import dataclass
from pydantic import Field
from textwrap import dedent
import arxiv





import feedparser
import requests
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET
import time





# Load environment variables
load_dotenv()
os.environ["SERPER_API_KEY"] = "3f056fb14a368c48a02a1109b88db19f2445ece2"
os.environ[
    "OPENAI_API_KEY"] = "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"
os.environ["OPENAI_MODEL"] = "gpt-4-32k"

# Streamlit App Configuration
st.set_page_config(
    page_title="AcademicGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(#b5e1ff 100%);
    }

    .main .block-container {
        background: rgba(219, 250, 210, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    .stSidebar {
        background: linear-gradient(180deg, #f2f5d0 0%, #fadad4 100%);
    }

    .stSidebar .sidebar-content {
        background: transparent;
    }

    h1 {
        color: #2C3E50;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h2 {
        color: #34495E;
        font-weight: 600;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }

    h3 {
        color: #2C3E50;
        font-weight: 500;
    }

    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    .stTextInput > div > div > input {
        border: 2px solid #E8E8E8;
        border-radius: 15px;
        padding: 0.8rem 1rem;
        transition: all 0.3s ease;
        font-size: 1rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }

    .stSelectbox > div > div > select {
        border: 2px solid #E8E8E8;
        border-radius: 15px;
        padding: 0.8rem 1rem;
    }

    .stRadio > div {
        background: rgba(102, 126, 234, 0.05);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }

    .stExpander {
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 15px;
        margin: 0.5rem 0;
        overflow: hidden;
    }

    .stExpander > div:first-child {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(102, 126, 234, 0.05);
        padding: 0.5rem;
        border-radius: 15px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #2C3E50;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
    }

    .success-box {
        background: linear-gradient(45deg, #00b894, #00a085);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }

    .info-box {
        background: linear-gradient(45deg, #0984e3, #74b9ff);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }

    .warning-box {
        background: linear-gradient(45deg, #fdcb6e, #e17055);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        display: block;
    }

    .metric-label {
        color: #2C3E50;
        font-weight: 500;
        margin-top: 0.5rem;
    }

    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(118, 75, 162, 0.05);
    }

    .sidebar-metric {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
    }

    .research-result {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }

    .chat-message {
        background: rgba(102, 126, 234, 0.05);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }

    .memory-entry {
        background: rgba(118, 75, 162, 0.05);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 4px solid #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Title with enhanced styling
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #f7f305;">🤖 AcademicGPT 🤖</h1>
    <p style="font-size: 1.2rem; color: #242023; font-weight: 400; margin-top: -1rem;">
        Advanced AI-powered research with memory and document intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory_context' not in st.session_state:
    st.session_state.memory_context = []
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


# PDF Generation Functions
def create_pdf_report(content, title, research_mode):
    """Create a professional PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)

    # Define styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#2C3E50'),
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=HexColor('#667eea'),
        fontName='Helvetica'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=HexColor('#2C3E50'),
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=10,
        alignment=TA_JUSTIFY,
        textColor=HexColor('#2C3E50'),
        fontName='Helvetica'
    )

    # Build story
    story = []

    # Title page
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Research Mode: {research_mode}", subtitle_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
    story.append(Spacer(1, 24))
    story.append(Paragraph("Generated by CrewAI Research Assistant", subtitle_style))
    story.append(PageBreak())

    # Content
    story.append(Paragraph("Research Report", heading_style))
    story.append(Spacer(1, 12))

    # Split content into paragraphs and format
    paragraphs = str(content).split('\n\n')
    for para in paragraphs:
        if para.strip():
            # Check if it's a heading (contains common heading patterns)
            if any(keyword in para.lower() for keyword in
                   ['abstract', 'introduction', 'methodology', 'conclusion', 'results', 'literature review',
                    'architecture']):
                story.append(Paragraph(para.strip(), heading_style))
            else:
                story.append(Paragraph(para.strip(), body_style))
            story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


# Document Processing Functions
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def extract_text_from_docx(docx_file):
    """Extract text from Word document"""
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return ""


def chunk_text(text, chunk_size=2000, overlap=400):
    """Split text into chunks for better embeddings"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_vector_database(documents_data):
    """Create FAISS vector database from documents"""
    if not documents_data:
        return None, []

    # Load embedding model
    model = load_embedding_model()

    all_chunks = []
    metadata = []

    for doc_name, text in documents_data:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.extend([{"source": doc_name, "chunk_id": i} for i in range(len(chunks))])

    if not all_chunks:
        return None, []

    # Create embeddings
    embeddings = model.encode(all_chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))

    return index, list(zip(all_chunks, metadata))


def search_documents(query, vector_db, documents, top_k=10):
    """Search documents using vector similarity"""
    if not vector_db or not documents:
        return []

    model = load_embedding_model()

    # Create query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = vector_db.search(query_embedding.astype(np.float32), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents) and idx >= 0:
            chunk, metadata = documents[idx]
            results.append({
                "text": chunk,
                "source": metadata["source"],
                "score": float(score),
                "chunk_id": metadata["chunk_id"]
            })

    return results


# Memory Management Functions
def add_to_memory(query, response, research_mode):
    """Add interaction to memory"""
    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response,
        "research_mode": research_mode,
        "session_id": st.session_state.current_session_id
    }
    st.session_state.memory_context.append(memory_entry)

    # Keep only last 10 interactions to manage memory
    if len(st.session_state.memory_context) > 15:
        st.session_state.memory_context = st.session_state.memory_context[-15:]


def get_memory_context():
    """Get formatted memory context for agents"""
    if not st.session_state.memory_context:
        return ""

    context = "Previous conversation context:\n"
    for entry in st.session_state.memory_context[-5:]:  # Last 5 interactions
        context += f"- Query: {entry['query'][:100]}...\n"
        context += f"  Mode: {entry['research_mode']}\n"
        context += f"  Time: {entry['timestamp'][:19]}\n\n"

    return context


# Enhanced Document Search Tool
class DocumentSearchTool(BaseTool):
    name: str = "document_search"
    description: str = "Search through uploaded documents to find relevant information based on semantic similarity"

    def __init__(self):
        super().__init__()

    def _run(self, query: str) -> str:
        """Execute the document search with improved error handling"""
        try:
            if not st.session_state.vector_db or not st.session_state.documents:
                return "No documents are currently available for search. Please upload documents first."

            results = search_documents(
                query,
                st.session_state.vector_db,
                st.session_state.documents,
                top_k=10
            )

            if not results:
                return f"No relevant information found in the uploaded documents for query: '{query}'"

            search_results = f"Found {len(results)} relevant document sections for '{query}':\n\n"

            for i, result in enumerate(results, 1):
                search_results += f"**Result {i}** (from {result['source']}, similarity: {result['score']:.3f}):\n"
                search_results += f"{result['text']}\n"
                search_results += "-" * 50 + "\n\n"

            return search_results

        except Exception as e:
            return f"Error searching documents: {str(e)}"


# Memory-aware Tool
class MemorySearchTool(BaseTool):
    name: str = "memory_search"
    description: str = "Search through conversation history and memory for context"

    def __init__(self):
        super().__init__()

    def _run(self, query: str) -> str:
        """Search through conversation memory"""
        if not st.session_state.memory_context:
            return "No previous conversation history available."

        # Simple keyword-based search through memory
        relevant_memories = []
        query_words = query.lower().split()

        for memory in st.session_state.memory_context:
            memory_text = (memory['query'] + ' ' + str(memory['response'])).lower()

            # Check if any query words are in the memory
            if any(word in memory_text for word in query_words):
                relevant_memories.append(memory)

        if not relevant_memories:
            return "No relevant information found in conversation history."

        result = "Relevant information from previous conversations:\n\n"
        for memory in relevant_memories[-3:]:  # Last 3 relevant memories
            result += f"Previous Query: {memory['query']}\n"
            result += f"Mode: {memory['research_mode']}\n"
            result += f"Time: {memory['timestamp'][:19]}\n"
            result += f"Response Summary: {str(memory['response'])[:200]}...\n\n"

        return result


# ===== CITATION DATA STRUCTURE =====
@dataclass
class Citation:
    """Data structure for academic citations"""
    authors: List[str]
    title: str
    journal: str
    year: int
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    citation_type: str = "conference"  # journal, conference, book, website


# ===== CITATION TOOL =====
class CitationTool(BaseTool):
    name: str = "Citation Manager"
    description: str = "Professional citation management and formatting tool for academic papers. Supports IEEE, APA, and MLA styles."

    # Declare fields for Pydantic model
    citations: Dict[int, Citation] = Field(default_factory=dict)
    citation_counter: int = Field(default=1)
    style: str = Field(default="ieee")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, action: str, **kwargs) -> str:
        """Main execution method for the citation tool"""
        if action == "add_citation":
            return self._add_citation_handler(**kwargs)
        elif action == "format_in_text":
            return self._format_in_text_handler(**kwargs)
        elif action == "generate_references":
            return self._generate_references_handler()
        elif action == "validate_citations":
            return self._validate_citations_handler()
        elif action == "set_style":
            return self._set_style_handler(**kwargs)
        else:
            return f"Unknown action: {action}. Available actions: add_citation, format_in_text, generate_references, validate_citations, set_style"

    def _add_citation_handler(self, authors: List[str], title: str, journal: str, year: int, **optional_fields) -> str:
        """Add a new citation and return its reference number"""
        try:
            citation = Citation(
                authors=authors,
                title=title,
                journal=journal,
                year=year,
                volume=optional_fields.get('volume'),
                issue=optional_fields.get('issue'),
                pages=optional_fields.get('pages'),
                doi=optional_fields.get('doi'),
                url=optional_fields.get('url'),
                citation_type=optional_fields.get('citation_type', 'journal')
            )

            ref_num = self.citation_counter
            self.citations[ref_num] = citation
            self.citation_counter += 1

            return f"Citation added successfully. Reference number: [{ref_num}]"
        except Exception as e:
            return f"Error adding citation: {str(e)}"

    def _format_in_text_handler(self, ref_nums: List[int]) -> str:
        """Format in-text citations [1], [1,2], [1-3]"""
        if not ref_nums:
            return ""

        ref_nums = sorted(ref_nums)
        if len(ref_nums) == 1:
            return f"[{ref_nums[0]}]"
        elif len(ref_nums) == 2:
            return f"[{ref_nums[0]},{ref_nums[1]}]"
        elif self._is_consecutive(ref_nums):
            return f"[{ref_nums[0]}-{ref_nums[-1]}]"
        else:
            return f"[{','.join(map(str, ref_nums))}]"

    def _generate_references_handler(self) -> str:
        """Generate formatted reference list"""
        if not self.citations:
            return "No citations available. Add citations first using the add_citation action."

        references = []
        for num in sorted(self.citations.keys()):
            citation = self.citations[num]
            formatted = self._format_single_reference(num, citation)
            references.append(formatted)

        return "\n".join(references)

    def _validate_citations_handler(self) -> str:
        """Validate citation completeness and format"""
        if not self.citations:
            return "No citations to validate."

        issues = []
        stats = {
            "total_citations": len(self.citations),
            "missing_doi": 0,
            "recent_papers": 0,
            "citation_types": {}
        }

        current_year = datetime.now().year

        for num, citation in self.citations.items():
            # Check for missing DOI
            if not citation.doi:
                stats["missing_doi"] += 1

            # Count recent papers (last 5 years)
            if current_year - citation.year <= 5:
                stats["recent_papers"] += 1

            # Count citation types
            if citation.citation_type not in stats["citation_types"]:
                stats["citation_types"][citation.citation_type] = 0
            stats["citation_types"][citation.citation_type] += 1

        validation_report = f"""
Citation Validation Report:
- Total Citations: {stats['total_citations']}
- Missing DOI: {stats['missing_doi']}
- Recent Papers (last 5 years): {stats['recent_papers']}
- Citation Types: {stats['citation_types']}
        """

        return validation_report.strip()

    def _set_style_handler(self, style: str) -> str:
        """Set citation style"""
        valid_styles = ["ieee", "apa", "mla"]
        if style.lower() in valid_styles:
            self.style = style.lower()
            return f"Citation style set to: {style.upper()}"
        else:
            return f"Invalid style. Valid styles are: {', '.join(valid_styles)}"

    def _format_single_reference(self, num: int, citation: Citation) -> str:
        """Format single reference based on citation style"""
        if self.style == "ieee":
            return self._format_ieee(num, citation)
        elif self.style == "apa":
            return self._format_apa(num, citation)
        else:
            return self._format_ieee(num, citation)

    def _format_ieee(self, num: int, citation: Citation) -> str:
        """Format citation in IEEE style"""
        authors_str = self._format_authors_ieee(citation.authors)

        if citation.citation_type == "journal":
            ref = f"[{num}] {authors_str}, \"{citation.title},\" {citation.journal}"
            if citation.volume:
                ref += f", vol. {citation.volume}"
            if citation.issue:
                ref += f", no. {citation.issue}"
            if citation.pages:
                ref += f", pp. {citation.pages}"
            ref += f", {citation.year}."
            if citation.doi:
                ref += f" DOI: {citation.doi}"

        elif citation.citation_type == "conference":
            ref = f"[{num}] {authors_str}, \"{citation.title},\" in {citation.journal}, {citation.year}"
            if citation.pages:
                ref += f", pp. {citation.pages}"
            ref += "."

        return ref

    def _format_apa(self, num: int, citation: Citation) -> str:
        """Format citation in APA style"""
        authors_str = self._format_authors_apa(citation.authors)
        ref = f"{authors_str} ({citation.year}). {citation.title}. {citation.journal}"
        if citation.volume and citation.issue:
            ref += f", {citation.volume}({citation.issue})"
        elif citation.volume:
            ref += f", {citation.volume}"
        if citation.pages:
            ref += f", {citation.pages}"
        ref += "."
        if citation.doi:
            ref += f" https://doi.org/{citation.doi}"
        return ref

    def _format_authors_ieee(self, authors: List[str]) -> str:
        """Format author names for IEEE style"""
        if len(authors) == 1:
            return authors[0]
        elif len(authors) <= 6:
            return ", ".join(authors[:-1]) + f", and {authors[-1]}"
        else:
            return ", ".join(authors[:6]) + ", et al."

    def _format_authors_apa(self, authors: List[str]) -> str:
        """Format author names for APA style"""
        if len(authors) == 1:
            return authors[0]
        elif len(authors) <= 20:
            return ", ".join(authors[:-1]) + f", & {authors[-1]}"
        else:
            return ", ".join(authors[:19]) + ", ... " + authors[-1]

    def _is_consecutive(self, nums: List[int]) -> bool:
        """Check if numbers are consecutive"""
        return all(nums[i] + 1 == nums[i + 1] for i in range(len(nums) - 1))


# ===== FORMATTING TOOL =====
class AcademicFormattingTool(BaseTool):
    name: str = "Academic Document Formatter"
    description: str = "Professional academic document formatting tool for creating structured research papers with proper sections, tables, and formatting."

    # Declare fields for Pydantic model
    document_structure: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "",
        "authors": [],
        "abstract": "",
        "sections": [],
        "references": "",
        "word_counts": {}
    })
    formatting_rules: Dict[str, Any] = Field(default_factory=lambda: {
        "line_spacing": "double",
        "font": "Times New Roman",
        "font_size": 12,
        "margins": "1 inch",
        "citation_style": "ieee"
    })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, action: str, **kwargs) -> str:
        """Main execution method for the formatting tool"""
        if action == "create_template":
            return self._create_template_handler(**kwargs)
        elif action == "format_table":
            return self._format_table_handler(**kwargs)
        elif action == "count_words":
            return self._count_words_handler(**kwargs)
        elif action == "validate_structure":
            return self._validate_structure_handler(**kwargs)
        elif action == "format_section":
            return self._format_section_handler(**kwargs)
        else:
            return f"Unknown action: {action}. Available actions: create_template, format_table, count_words, validate_structure, format_section"

    def _create_template_handler(self, title: str, authors: List[str]) -> str:
        """Create formatted academic document template"""
        template = f"""# {title}

**Authors:** {', '.join(authors)}

---

## Abstract


**Keywords:** [5-7 keywords]

---

## 1. Introduction


---

## 2. Related Work


### Table 1: Comparative Analysis of Related Work

| Paper Title | Authors | Year | Method/Algorithm | Accuracy | Key Findings | Limitations |
|-------------|---------|------|------------------|----------|--------------|-------------|
| | | | | | | |

---

## 3. Comparative Analysis and Results


---

## 4. Current Work and Future Directions


### 4.1 Current State of Research


### 4.2 Emerging Trends and Technologies


### 4.3 Future Research Directions


---

## 5. Conclusion


---

## References


---

**Document Statistics:**
- Target Length: 5000-6000 words
- Sections: 5 main sections
- Tables: 1 comparative analysis table
- References: Minimum 15
"""
        return template

    def _format_table_handler(self, headers: List[str], rows: List[List[str]], caption: str = "",
                              table_num: int = 1) -> str:
        """Format academic table with proper structure"""
        # Create header row
        header_row = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join([" --- " for _ in headers]) + "|"

        # Create data rows
        data_rows = []
        for row in rows:
            data_row = "| " + " | ".join(row) + " |"
            data_rows.append(data_row)

        # Combine table elements
        table = [f"### Table {table_num}: {caption}", "", header_row, separator] + data_rows + [""]

        return "\n".join(table)

    def _count_words_handler(self, text: str) -> str:
        """Count words in different sections of the document"""
        # Remove markdown formatting for accurate word count
        clean_text = re.sub(r'[#*`\[\](){}]', '', text)
        clean_text = re.sub(r'\|.*\|', '', clean_text)  # Remove table content

        total_words = len(clean_text.split())

        # Extract section word counts (simplified)
        sections = {
            "abstract": self._extract_section_words(text, "Abstract"),
            "introduction": self._extract_section_words(text, "Introduction"),
            "related_work": self._extract_section_words(text, "Related Work"),
            "analysis": self._extract_section_words(text, "Comparative Analysis"),
            "current_work": self._extract_section_words(text, "Current Work"),
            "conclusion": self._extract_section_words(text, "Conclusion")
        }

        result = f"""Word Count Analysis:
- Total Words: {total_words}
- Abstract: {sections['abstract']} words
- Introduction: {sections['introduction']} words
- Related Work: {sections['related_work']} words
- Analysis: {sections['analysis']} words
- Current Work: {sections['current_work']} words
- Conclusion: {sections['conclusion']} words
"""
        return result

    def _validate_structure_handler(self, document: str) -> str:
        """Validate document structure and formatting"""
        required_sections = {
            "Abstract": "## Abstract" in document,
            "Introduction": "## 1. Introduction" in document or "## Introduction" in document,
            "Related Work": "Related Work" in document,
            "Analysis": "Analysis" in document or "Results" in document,
            "Current Work": "Current Work" in document or "Future" in document,
            "Conclusion": "## Conclusion" in document or "## 5. " in document,
            "References": "## References" in document
        }

        word_counts = self._count_words_simple(document)
        issues = []

        # Check required sections
        missing_sections = [section for section, present in required_sections.items() if not present]
        if missing_sections:
            issues.append(f"Missing sections: {', '.join(missing_sections)}")

        # Check word count compliance
        if word_counts < 3000 or word_counts > 4000:
            issues.append(f"Document length ({word_counts} words) outside target range (3000-4000)")

        if issues:
            return f"Validation Issues Found:\n" + "\n".join(f"- {issue}" for issue in issues)
        else:
            return "Document structure validation passed successfully!"

    def _format_section_handler(self, level: int, title: str, number: str = None) -> str:
        """Format section headers with proper hierarchy"""
        if level == 1:
            prefix = f"{number}. " if number else ""
            return f"## {prefix}{title}\n"
        elif level == 2:
            prefix = f"{number} " if number else ""
            return f"### {prefix}{title}\n"
        elif level == 3:
            prefix = f"{number} " if number else ""
            return f"#### {prefix}{title}\n"
        else:
            return f"**{title}**\n"

    def _extract_section_words(self, text: str, section_name: str) -> int:
        """Extract word count for specific section"""
        pattern = rf"## \d*\.?\s*{section_name}.*?(?=## |\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            section_text = re.sub(r'[#*`\[\](){}]', '', match.group())
            return len(section_text.split())
        return 0

    def _count_words_simple(self, text: str) -> int:
        """Simple word count for validation"""
        clean_text = re.sub(r'[#*`\[\](){}]', '', text)
        clean_text = re.sub(r'\|.*\|', '', clean_text)
        return len(clean_text.split())


# ===== ANALYSIS TOOL =====
class ResearchAnalysisTool(BaseTool):
    name: str = "Research Analysis Tool"
    description: str = "Advanced research data analysis and comparison tool for academic papers. Analyzes performance metrics, identifies research gaps, and generates comprehensive summaries."

    # Declare fields for Pydantic model
    research_data: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_cache: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, action: str, **kwargs) -> str:
        """Main execution method for the analysis tool"""
        if action == "add_research":
            return self._add_research_handler(**kwargs)
        elif action == "performance_comparison":
            return self._performance_comparison_handler()
        elif action == "generate_comparison_table":
            return self._generate_comparison_table_handler()
        elif action == "identify_gaps":
            return self._identify_gaps_handler()
        elif action == "analysis_summary":
            return self._analysis_summary_handler()
        elif action == "clear_data":
            return self._clear_data_handler()
        else:
            return f"Unknown action: {action}. Available actions: add_research, performance_comparison, generate_comparison_table, identify_gaps, analysis_summary, clear_data"

    def _add_research_handler(self, title: str, authors: List[str], year: int, method: str, accuracy: Any,
                              **kwargs) -> str:
        """Add research paper data for analysis"""
        try:
            data = {
                "title": title,
                "authors": authors,
                "year": year,
                "method": method,
                "accuracy": accuracy,
                "key_findings": kwargs.get("key_findings", ""),
                "limitations": kwargs.get("limitations", ""),
                "dataset": kwargs.get("dataset", ""),
                "domain": kwargs.get("domain", "")
            }

            self.research_data.append(data)
            return f"Research entry added successfully. Total entries: {len(self.research_data)}"
        except Exception as e:
            return f"Error adding research entry: {str(e)}"

    def _performance_comparison_handler(self) -> str:
        """Analyze and compare performance metrics across papers"""
        if not self.research_data:
            return "No research data available. Add research entries first using the add_research action."

        # Extract performance metrics
        accuracies = []
        methods = {}
        yearly_progress = {}

        for entry in self.research_data:
            # Collect accuracy data
            if isinstance(entry["accuracy"], (int, float)):
                accuracies.append(entry["accuracy"])
            elif isinstance(entry["accuracy"], str):
                # Try to extract numeric value from string
                numeric_match = re.search(r'(\d+\.?\d*)%?', entry["accuracy"])
                if numeric_match:
                    accuracies.append(float(numeric_match.group(1)))

            # Group by method
            method = entry.get("method", "Unknown")
            if method not in methods:
                methods[method] = []
            methods[method].append(entry)

            # Track yearly progress
            year = entry.get("year", 2020)
            if year not in yearly_progress:
                yearly_progress[year] = []
            yearly_progress[year].append(entry)

        # Calculate statistics
        performance_report = f"""Performance Comparison Analysis:

Total Papers Analyzed: {len(self.research_data)}

Performance Statistics:
- Mean Accuracy: {sum(accuracies) / len(accuracies):.2f}% (based on {len(accuracies)} papers)
- Maximum Accuracy: {max(accuracies):.2f}%
- Minimum Accuracy: {min(accuracies):.2f}%
- Accuracy Range: {max(accuracies) - min(accuracies):.2f}%

Method Distribution:
{self._format_method_distribution(methods)}

Yearly Trends:
{self._format_yearly_trends(yearly_progress)}
"""
        return performance_report

    def _generate_comparison_table_handler(self) -> str:
        """Generate structured comparison table data"""
        if not self.research_data:
            return "No research data available for table generation."

        # Create table headers
        headers = ["Paper Title", "Authors", "Year", "Method/Algorithm", "Accuracy", "Key Findings", "Limitations"]

        # Create table rows
        table_rows = []
        for entry in sorted(self.research_data, key=lambda x: x.get("year", 0), reverse=True):
            row = [
                entry.get("title", "N/A")[:50] + ("..." if len(entry.get("title", "")) > 50 else ""),
                ", ".join(entry.get("authors", [])[:2]) + (" et al." if len(entry.get("authors", [])) > 2 else ""),
                str(entry.get("year", "N/A")),
                entry.get("method", "N/A"),
                str(entry.get("accuracy", "N/A")),
                entry.get("key_findings", "N/A")[:100] + ("..." if len(entry.get("key_findings", "")) > 100 else ""),
                entry.get("limitations", "N/A")[:80] + ("..." if len(entry.get("limitations", "")) > 80 else "")
            ]
            table_rows.append(row)

        # Format as markdown table
        header_row = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join([" --- " for _ in headers]) + "|"

        data_rows = []
        for row in table_rows:
            data_row = "| " + " | ".join(row) + " |"
            data_rows.append(data_row)

        table = [header_row, separator] + data_rows
        return "\n".join(table)

    def _identify_gaps_handler(self) -> str:
        """Identify potential research gaps and opportunities"""
        if not self.research_data:
            return "No research data available for gap analysis."

        gaps = []

        # Analyze method diversity
        methods = set(entry.get("method") for entry in self.research_data)
        if len(methods) < 5:
            gaps.append("Limited diversity in methodological approaches")

        # Check for performance plateaus
        recent_accuracies = [
            entry.get("accuracy") for entry in self.research_data
            if entry.get("year", 0) >= 2020 and isinstance(entry.get("accuracy"), (int, float))
        ]
        if recent_accuracies and max(recent_accuracies) - min(recent_accuracies) < 5:
            gaps.append("Performance improvements have plateaued in recent years")

        # Identify temporal gaps
        years = [entry.get("year") for entry in self.research_data if entry.get("year")]
        if years:
            sorted_years = sorted(set(years))
            year_gaps = []
            for i in range(len(sorted_years) - 1):
                if sorted_years[i + 1] - sorted_years[i] > 2:
                    year_gaps.append(f"{sorted_years[i]}-{sorted_years[i + 1]}")
            if year_gaps:
                gaps.append(f"Research gaps in years: {', '.join(year_gaps)}")

        if gaps:
            return "Identified Research Gaps:\n" + "\n".join(f"- {gap}" for gap in gaps)
        else:
            return "No significant research gaps identified based on current data."

    def _analysis_summary_handler(self) -> str:
        """Generate comprehensive analysis summary"""
        if not self.research_data:
            return "No research data available for summary generation."

        # Get performance analysis
        performance_data = self._get_performance_stats()

        # Get top performers
        top_papers = self._get_top_performers(3)

        summary = f"""## Research Analysis Summary

### Overview
- **Total Papers Analyzed:** {len(self.research_data)}
- **Date Range:** {min(entry.get('year', 2020) for entry in self.research_data)} - {max(entry.get('year', 2024) for entry in self.research_data)}

### Performance Insights
{performance_data}

### Top Performing Approaches
{self._format_top_performers(top_papers)}

### Research Landscape
- **Unique Methods:** {len(set(entry.get('method', 'Unknown') for entry in self.research_data))}
- **Recent Papers (2020+):** {len([p for p in self.research_data if p.get('year', 0) >= 2020])}

### Recommendations
- Focus on underexplored methodological approaches
- Address identified performance plateaus
- Consider cross-validation of top-performing methods
"""
        return summary

    def _clear_data_handler(self) -> str:
        """Clear all research data"""
        count = len(self.research_data)
        self.research_data.clear()
        return f"Cleared {count} research entries. Data reset successfully."

    def _get_performance_stats(self) -> str:
        """Get performance statistics"""
        accuracies = []
        for entry in self.research_data:
            if isinstance(entry.get("accuracy"), (int, float)):
                accuracies.append(entry["accuracy"])

        if accuracies:
            return f"""- **Average Performance:** {sum(accuracies) / len(accuracies):.2f}%
- **Best Performance:** {max(accuracies):.2f}%
- **Performance Range:** {max(accuracies) - min(accuracies):.2f}%"""
        else:
            return "- No numeric performance data available"

    def _get_top_performers(self, n: int) -> List[Dict]:
        """Get top N performing papers"""
        papers_with_accuracy = [
            p for p in self.research_data
            if isinstance(p.get("accuracy"), (int, float))
        ]

        return sorted(
            papers_with_accuracy,
            key=lambda x: x.get("accuracy", 0),
            reverse=True
        )[:n]

    def _format_method_distribution(self, methods: Dict) -> str:
        """Format method distribution"""
        formatted = []
        total = sum(len(papers) for papers in methods.values())
        for method, papers in sorted(methods.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(papers)
            percentage = (count / total) * 100
            formatted.append(f"- {method}: {count} papers ({percentage:.1f}%)")
        return "\n".join(formatted)

    def _format_yearly_trends(self, yearly_data: Dict) -> str:
        """Format yearly trends"""
        formatted = []
        for year in sorted(yearly_data.keys()):
            papers = yearly_data[year]
            accuracies = [p.get("accuracy") for p in papers if isinstance(p.get("accuracy"), (int, float))]
            avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0
            formatted.append(f"- {year}: {len(papers)} papers, avg accuracy: {avg_acc:.1f}%")
        return "\n".join(formatted)

    def _format_top_performers(self, top_papers: List[Dict]) -> str:
        """Format top performers"""
        if not top_papers:
            return "- No performance data available"

        formatted = []
        for i, paper in enumerate(top_papers, 1):
            title = paper.get("title", "Unknown")[:50] + ("..." if len(paper.get("title", "")) > 50 else "")
            accuracy = paper.get("accuracy", "N/A")
            method = paper.get("method", "N/A")
            formatted.append(f"{i}. **{title}** - {accuracy}% ({method})")

        return "\n".join(formatted)


#-----------------------------Arxiv search tool---------------------------#
class ArxivSearchInput(BaseModel):
    """Input schema for ArxivSearchTool."""
    query: str = Field(..., description="Search query for arXiv papers")
    max_results: int = Field(default=15, description="Maximum number of results to return")


class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = "Searches arXiv for academic papers related to the query. Returns title, authors, summary, and PDF URL."
    args_schema: Type[BaseModel] = ArxivSearchInput

    def _run(self, query: str, max_results: int = 15) -> str:
        """Execute the arXiv search."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in search.results():
                paper_info = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary[:300] + "..." if len(result.summary) > 300 else result.summary,
                    "pdf_url": result.pdf_url
                }
                papers.append(paper_info)

            # Format the results as a readable string
            formatted_results = []
            for i, paper in enumerate(papers, 1):
                authors_str = ", ".join(paper["authors"][:3])  # Show first 3 authors
                if len(paper["authors"]) > 3:
                    authors_str += " et al."

                formatted_paper = f"""
Paper {i}:
📘 Title: {paper["title"]}
👥 Authors: {authors_str}
📝 Summary: {paper["summary"]}
🔗 PDF URL: {paper["pdf_url"]}
{"=" * 50}
"""
                formatted_results.append(formatted_paper)

            return "\n".join(formatted_results) if formatted_results else "No papers found for the given query."

        except Exception as e:
            return f"Error searching arXiv: {str(e)}"


# Alternative: Simple function-based tool (if you prefer)
def create_arxiv_search_tool():
    """Create an arXiv search tool using the tool decorator."""
    from crewai_tools import tool

    @tool("arxiv_search")
    def arxiv_search(query: str, max_results: int = 15) -> str:
        """
        Searches arXiv for academic papers related to the query.

        Args:
            query (str): Search query for arXiv papers
            max_results (int): Maximum number of results to return (default: 15)

        Returns:
            str: Formatted string containing paper details
        """
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )

            papers = []
            for result in search.results():
                paper_info = {
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary[:300] + "..." if len(result.summary) > 300 else result.summary,
                    "pdf_url": result.pdf_url
                }
                papers.append(paper_info)

            # Format the results as a readable string
            formatted_results = []
            for i, paper in enumerate(papers, 1):
                authors_str = ", ".join(paper["authors"][:3])  # Show first 3 authors
                if len(paper["authors"]) > 3:
                    authors_str += " et al."

                formatted_paper = f"""
Paper {i}:
📘 Title: {paper["title"]}
👥 Authors: {authors_str}
📝 Summary: {paper["summary"]}
🔗 PDF URL: {paper["pdf_url"]}
{"=" * 50}
"""
                formatted_results.append(formatted_paper)

            return "\n".join(formatted_results) if formatted_results else "No papers found for the given query."

        except Exception as e:
            return f"Error searching arXiv: {str(e)}"

    return arxiv_search


# Tools
search_tool = SerperDevTool()
doc_search_tool = DocumentSearchTool()
memory_tool = MemorySearchTool()
citation = CitationTool()
academic_formatting= AcademicFormattingTool()
research_analysis_tool= ResearchAnalysisTool()
arxiv_tool = ArxivSearchTool()




# Enhanced Agents with Memory
def create_agents(use_documents=False):
    memory_context = get_memory_context()

    if use_documents:
        # Document-based researcher with memory
        researcher = Agent(
            role='Document Researcher with Memory',
            goal='Extract relevant information from uploaded documents about {topic} while considering previous conversation context',
            verbose=True,
            backstory=(
                "You are an expert document analyst with perfect memory of previous conversations. "
                "You analyze uploaded documents and remember all previous interactions to provide contextual responses. "
                f"Previous context: {memory_context}"
            ),
            tools=[doc_search_tool, memory_tool],
            allow_delegation=True
        )

        writer = Agent(
            role='Document-based Writer with Memory',
            goal='Create comprehensive reports based on document analysis about {topic} considering conversation history',
            verbose=True,
            backstory=(
                "You specialize in synthesizing information from documents and previous conversations into well-structured reports. "
                "You remember all previous interactions and build upon them for continuity. "
                f"Previous context: {memory_context}"
            ),
            tools=[doc_search_tool, memory_tool, citation, academic_formatting, research_analysis_tool ],
            allow_delegation=False
        )
    else:
        # Internet-based researcher with memory
        researcher = Agent(
            role='Senior Research Analyst & Intelligence Specialist',
            goal=(
                'Execute comprehensive research investigations on {topic} leveraging advanced web intelligence '
                'and systematic data analysis to deliver actionable insights and evidence-based findings'
                'producing 5000-6000 word for your research'
            ),
            verbose=True,
            backstory=(
                "You are an experienced research analyst with expertise in digital intelligence gathering, "
                "data synthesis, and strategic information analysis. Your methodology combines systematic "
                "web research with contextual memory retention to build comprehensive knowledge bases. "
                "You excel at identifying reliable sources, cross-referencing information, and presenting "
                "findings in structured, actionable formats. Your research approach emphasizes accuracy, "
                "thoroughness, and continuous knowledge building through iterative investigation cycles. "
                f"Research Context & Prior Findings: {memory_context}"
            ),
            tools=[
                search_tool,  # Web search and information retrieval
                memory_tool,  # Knowledge persistence and context management
                arxiv_tool
            ],
            allow_delegation=True,
            max_iter=10,  # Maximum iterations for complex research tasks
            max_execution_time=300,  # Timeout for research operations
            system_message=(
                "As a Senior Research Analyst, you must:\n"
                "1. Conduct systematic and comprehensive research using available tools\n"
                "2. Validate information through multiple credible sources\n"
                "3. Maintain detailed records of all findings and sources\n"
                "4. Build upon previous research context and avoid redundant searches\n"
                "5. Present findings in clear, structured, and actionable formats\n"
                "6. Identify knowledge gaps and recommend further investigation areas\n"
                "7. Ensure all research adheres to ethical guidelines and best practices"
            )
        )

        writer = Agent(
            role='Senior Academic Research Writer & Publication Specialist',
            goal=(
                'Compose comprehensive academic research papers on {topic} following rigorous scholarly standards, '
                'producing stricly 5000-6000 words publications with proper citations not less than that, structured methodology, and '
                'evidence-based analysis that advances knowledge in the field'
            ),
            verbose=True,
            backstory=(
                "You are an accomplished academic writer and research publication specialist with extensive "
                "experience in scholarly communication, peer-reviewed publication standards, and systematic "
                "literature review methodologies. Your expertise spans technical writing, citation management, "
                "comparative analysis, and structured academic formatting. You excel at synthesizing complex "
                "research findings into coherent narratives while maintaining rigorous academic standards. "
                "Your writing consistently demonstrates critical thinking, methodological rigor, and clear "
                "communication of technical concepts to both specialist and general academic audiences. "
                f"Research Foundation & Historical Context: {memory_context}"
            ),
            tools=[
                search_tool,  # Academic database and literature search
                memory_tool,  # Knowledge base and context management
                citation,  # Reference management and formatting
                research_analysis_tool,    # Statistical and comparative analysis
                academic_formatting,  # Document structure and academic formatting
                arxiv_tool
            ],
            allow_delegation=False,
            max_iter=15,
            max_execution_time=600,
            system_message=(
                "As a Senior Academic Research Writer, you must produce research papers with the following "
                "mandatory structure and specifications:\n\n"

                "DOCUMENT STRUCTURE & WORD COUNTS:\n"
                "• Total Length: 5000-6000 words\n"
                "• Abstract: 150-200 words (structured: background, methods, results, conclusions)\n"
                "• Introduction: 300-400 words (context, problem statement, objectives, contributions)\n"
                "• Related Work: 400-500 words (literature review in structured table format)\n"
                "• Comparative Analysis: 300-400 words (results comparison and critical evaluation)\n"
                "• Current Work & Future Directions: 1000-1500 words (detailed analysis and projections)\n"
                "• Conclusion: 300 words (summary, implications, recommendations)\n"
                "• References: Minimum 15 scholarly references (IEEE/APA format)\n\n"

                "RELATED WORK TABLE FORMAT:\n"
                "| Paper Title | Authors | Year | Method/Algorithm/Architecture | Accuracy/Performance | Key Findings | Limitations |\n\n"

                "CITATION REQUIREMENTS:\n"
                "• Every paragraph must contain at least one in-text citation [1-15]\n"
                "• Use numerical citation system: [1], [2], [1,3], [1-5]\n"
                "• Ensure citations support all claims and statements\n"
                "• Cross-reference table entries with numbered citations\n\n"

                "ACADEMIC WRITING STANDARDS:\n"
                "• Use formal, objective, third-person academic tone\n"
                "• Employ precise technical terminology consistently\n"
                "• Structure paragraphs with topic sentences and logical flow\n"
                "• Include quantitative data and performance metrics where available\n"
                "• Maintain coherent argumentation throughout sections\n"
                "• Ensure smooth transitions between sections and paragraphs\n\n"

                "QUALITY ASSURANCE:\n"
                "• Verify all statistical data and performance figures\n"
                "• Ensure chronological accuracy of cited works\n"
                "• Maintain consistency in terminology and notation\n"
                "• Cross-check reference list against in-text citations\n"
                "• Validate technical accuracy of described methods\n\n"

                "RESEARCH METHODOLOGY:\n"
                "• Conduct systematic literature search using academic databases\n"
                "• Prioritize peer-reviewed sources and authoritative publications\n"
                "• Include recent developments (last 5 years) alongside foundational work\n"
                "• Analyze methodological approaches and comparative performance\n"
                "• Identify research gaps and future investigation opportunities\n\n"

                "OUTPUT REQUIREMENTS:\n"
                "• Generate publication-ready manuscript with professional formatting\n"
                "• Include section headers, subsections, and logical organization\n"
                "• Provide comprehensive reference list with complete bibliographic details\n"
                "• Ensure document meets specified word count targets for each section\n"
                "• Maintain academic integrity and original analysis throughout"
            ),
            additional_instructions=(
                "CRITICAL WRITING PROTOCOLS:\n"
                "1. Begin each writing session by reviewing previous context and maintaining continuity\n"
                "2. Structure the Related Work section as a comprehensive comparison table\n"
                "3. Ensure every major claim is supported by appropriate citations\n"
                "4. Maintain consistent academic voice and professional tone throughout\n"
                "5. Include quantitative performance metrics and statistical comparisons\n"
                "6. Conclude each section with clear transitions to subsequent sections\n"
                "7. Verify all technical terminology and methodological descriptions\n"
                "8. Generate a complete, publication-ready manuscript meeting all specifications"
            )
        )

    return researcher, writer


# Enhanced Tasks with Memory
def create_tasks(topic, use_documents=False):
    researcher, writer = create_agents(use_documents)

    if use_documents:
        research_task = Task(
            description=(
                f"Conduct comprehensive analysis of all uploaded documents related to {topic}. "
                f"Your responsibilities include:\n"
                f"1. Extract and catalog all relevant information from each uploaded document\n"
                f"2. Identify key themes, patterns, and insights across all documents\n"
                f"3. Note methodologies, findings, conclusions, and data points\n"
                f"4. Cross-reference information between documents to find connections\n"
                f"5. Identify gaps in the literature or conflicting viewpoints\n"
                f"6. Organize findings by relevance and importance to {topic}\n"
                f"7. Prepare detailed summaries of each document's contribution\n"
                f"8. Consider previous conversation history and build upon existing research context\n"
                f"9. Create a comprehensive knowledge base from all document sources\n"

            ),
            expected_output=(
                "A comprehensive research analysis report (1500-2000 words) containing:\n"
                "- Executive summary of all documents analyzed\n"
                "- Detailed findings organized by themes\n"
                "- Key insights and patterns identified\n"
                "- Cross-document connections and correlations\n"
                "- Literature gaps and research opportunities\n"
                "- Preliminary conclusions and recommendations\n"
                "- Source catalog with document summaries"
            ),
            agent=researcher,

        )

        write_task = Task(
            description=(
                f"Based on the comprehensive document analysis about {topic}, create a detailed academic-style report. "
                f"Your responsibilities include:\n"
                f"1. Synthesize all research findings into a cohesive narrative\n"
                f"2. Structure content following academic standards\n"
                f"3. Ensure the report reaches strictly 6000-7000 words in length not less than that\n"
                f"4. Maintain high academic rigor and professional tone\n"
                f"5. Include comprehensive citations and references\n"
                f"6. Build upon previous conversation history and research context\n"
                f"7. Provide detailed analysis rather than just summarization\n"
                f"8. Include critical evaluation of sources and methodologies\n"
                f"9. Offer original insights and recommendations\n"
                f"10. Ensure logical flow between all sections"
            ),
            expected_output=(
                "A comprehensive academic report (6000-7000 words) with the following structure:\n"
                "1. ABSTRACT (300-400 words)\n"
                "   - Research objectives and scope\n"
                "   - Key methodologies employed\n"
                "   - Major findings and conclusions\n"
                "   - Implications and recommendations\n\n"
                "2. INTRODUCTION (800-1000 words)\n"
                "   - Background and context of the topic\n"
                "   - Problem statement and research questions\n"
                "   - Scope and limitations\n"
                "   - Report structure overview\n\n"
                "3. METHODOLOGY (600-800 words)\n"
                "   - Document analysis approach\n"
                "   - Data extraction methods\n"
                "   - Analysis frameworks used\n"
                "   - Quality assessment criteria\n\n"
                "4. LITERATURE REVIEW (1500-2000 words)\n"
                "   - Comprehensive review of all source documents\n"
                "   - Thematic analysis of existing research\n"
                "   - Identification of research gaps\n"
                "   - Theoretical frameworks identified\n\n"
                "5. RESULTS AND FINDINGS (2000-2500 words)\n"
                "   - Detailed presentation of key findings\n"
                "   - Cross-document analysis and correlations\n"
                "   - Data synthesis and pattern identification\n"
                "   - Critical evaluation of evidence\n\n"
                "6. DISCUSSION (800-1000 words)\n"
                "   - Interpretation of findings\n"
                "   - Implications for theory and practice\n"
                "   - Limitations and considerations\n"
                "   - Future research directions\n\n"
                "7. CONCLUSION (400-500 words)\n"
                "   - Summary of major contributions\n"
                "   - Key recommendations\n"
                "   - Final observations\n\n"
                "8. REFERENCES AND CITATIONS\n"
                "   - Complete bibliography of all sources\n"
                "   - Proper academic citation format\n"
                "   - Document source attribution"
            ),
            agent=writer,
            context=[research_task],
            async_execution=False,

        )
    else:
        # Professional Research Task Template
        from textwrap import dedent

        # Main Research Task - Simplified and Error-Free
        research_task = Task(
            description=dedent("""
                Execute comprehensive internet-based research investigation on {topic} leveraging 
                advanced web search capabilities and systematic data analysis methodologies. 

                Your research mandate includes:

                CORE RESEARCH OBJECTIVES:
                - Conduct systematic web search across multiple authoritative sources
                - Identify and analyze current market trends, developments, and industry dynamics
                - Gather latest news, updates, and emerging patterns in the field
                - Cross-reference information across credible sources for validation
                - Synthesize findings into coherent, evidence-based insights

                CONTEXTUAL INTEGRATION REQUIREMENTS:
                - Review and incorporate previous conversation history and research context
                - Build upon existing knowledge base stored in memory systems
                - Identify knowledge gaps from previous research iterations
                - Connect new findings to established research threads
                - Demonstrate progressive understanding and insight evolution

                RESEARCH METHODOLOGY:
                - Utilize web search tools systematically for comprehensive coverage
                - Prioritize authoritative sources including industry reports, academic publications, and expert analysis
                - Focus on information published within the last 12-24 months for currency
                - Validate claims through multiple independent source verification
                - Document source credibility and potential bias considerations

                ANALYTICAL FRAMEWORK:
                - Identify key stakeholders, market players, and industry influencers
                - Analyze technological, economic, social, and regulatory factors
                - Examine implementation challenges, opportunities, and success factors
                - Assess future implications and strategic considerations
                - Synthesize actionable insights for decision-making purposes
            """),

            expected_output=dedent("""
                A comprehensive, professionally-structured research report containing exactly 
                3 substantive paragraphs that demonstrates advanced analytical synthesis:

                PARAGRAPH 1 - CURRENT LANDSCAPE & TRENDS (500-600 words):
                Present a comprehensive overview of the current state of the research topic, 
                incorporating latest market data, industry developments, and emerging trends. 
                This section should establish the foundational context while highlighting 
                recent developments that build upon previous research findings. Include 
                specific statistics, market figures, and credible source citations to 
                support analytical claims.

                PARAGRAPH 2 - DEEP ANALYSIS & INSIGHTS (400-500 words):
                Provide in-depth analytical examination of key findings, challenges, and 
                opportunities identified through web research. This section should demonstrate 
                critical thinking by connecting disparate information sources, identifying 
                patterns, and revealing insights that extend beyond surface-level observations. 
                Address implementation considerations, stakeholder perspectives, and strategic 
                implications while building upon previous research context.

                PARAGRAPH 3 - FUTURE PROJECTIONS & STRATEGIC IMPLICATIONS (300-400 words):
                Synthesize research findings into forward-looking analysis that projects 
                future developments, identifies strategic opportunities, and provides 
                actionable recommendations. This section should demonstrate how current 
                research builds upon and evolves previous understanding, offering enhanced 
                perspectives on potential outcomes, risk factors, and strategic considerations 
                for relevant stakeholders.

                QUALITY STANDARDS:
                - Each paragraph must be substantive, well-researched, and analytically rigorous
                - Include specific data points, statistics, and credible source references
                - Maintain professional tone suitable for executive-level consumption
                - Demonstrate clear progression of ideas and logical flow between sections
                - Showcase evolution of understanding from previous research iterations
                - Provide actionable insights and practical implications
            """),

            agent=researcher,
            async_execution=False
        )

        # Alternative Detailed Research Task

        detailed_research_task = Task(
            description=dedent("""
                Conduct comprehensive internet research investigation on {topic} utilizing 
                advanced web search methodologies and systematic analytical frameworks.

                RESEARCH EXECUTION REQUIREMENTS:
                - Deploy web search tools strategically across authoritative information sources
                - Analyze current industry trends, market dynamics, and recent developments
                - Integrate findings with previous research context and conversation history
                - Validate information through cross-source verification and credibility assessment
                - Synthesize insights that advance understanding beyond previous research iterations

                ANALYTICAL FOCUS AREAS:
                - Market landscape analysis and competitive positioning
                - Technological developments and innovation trends
                - Regulatory environment and policy implications
                - Implementation challenges and success factors
                - Future projections and strategic opportunities

                QUALITY ASSURANCE PROTOCOLS:
                - Prioritize sources published within last 18 months for currency
                - Cross-reference major claims across minimum 3 independent sources
                - Evaluate source authority, methodology, and potential bias
                - Document research trail for transparency and verification
                - Connect findings to previous research for demonstrated progression
            """),

            expected_output=dedent("""
                Professional research deliverable structured as 3 comprehensive paragraphs 
                (minimum 4500-6000 total words) demonstrating advanced research and analytical capabilities:

                **Executive Summary Paragraph**: Current state analysis incorporating latest 
                data, trends, and developments with clear connection to previous research context

                **Strategic Analysis Paragraph**: Deep-dive examination of key insights, 
                challenges, opportunities, and implementation considerations based on 
                comprehensive web research findings

                **Future Outlook Paragraph**: Forward-looking synthesis providing strategic 
                implications, projections, and actionable recommendations that build upon 
                and advance previous research understanding

                Each paragraph must include specific data points, credible citations, and 
                demonstrate professional-grade analytical rigor suitable for strategic 
                decision-making purposes.
            """),

            agent=researcher,
            async_execution=False
        )

        # Quick Research Variant
        rapid_research_task = Task(
            description=dedent("""
                Execute focused internet research on {topic} prioritizing recent developments 
                and current trends while building upon existing research context.

                RAPID RESEARCH PROTOCOL:
                - Conduct targeted web searches focusing on latest 6-12 months of developments
                - Identify 3-5 key trends or developments that advance current understanding
                - Cross-reference findings with previous research to show knowledge evolution
                - Prioritize actionable insights and strategic implications
                - Maintain research quality while optimizing for time efficiency
            """),

            expected_output=dedent("""
                Concise 3-paragraph research summary (4000-5000 words total) providing:

                1. **Current Status**: Latest developments and trends with key data points
                2. **Key Insights**: Critical analysis of findings and their implications  
                3. **Strategic Implications**: Forward-looking recommendations building on research evolution
                

                Focus on high-impact insights that demonstrate clear advancement from previous research.
            """),

            agent=researcher,
            async_execution=False
        )



        write_task = Task(
            description=(
                "Conduct comprehensive academic research and compose a scholarly publication on {topic} "
                "that synthesizes internet research findings with existing knowledge base and conversation history. "
                "Execute systematic literature analysis, methodological evaluation, and evidence-based synthesis "
                "to produce a publication-ready manuscript that advances understanding in the field. "
                "Integrate previous research insights while emphasizing novel findings from current web-based "
                "investigation to ensure continuity with established conversation context and build upon "
                "previously discussed theoretical frameworks and empirical findings."
            ),

            expected_output=(
                "A rigorously structured academic research paper (4000-5000 words) comprising the following "
                "mandatory components with specified word allocations:\n\n"

                "DOCUMENT STRUCTURE & SPECIFICATIONS:\n"
                "• ABSTRACT (400-550 words): Structured summary including background context, research "
                "  methodology, key findings, performance metrics, and significant conclusions\n"
                "• INTRODUCTION (400-500 words): Comprehensive problem contextualization, research gap "
                "  identification, objective formulation, and contribution statement with historical perspective\n"
                "• RELATED WORK & LITERATURE REVIEW (600-800 words): Systematic analysis presented in "
                "  structured comparison table format with narrative synthesis of methodological approaches, "
                "  performance benchmarks, and critical evaluation of existing research paradigms"
                "  In this table atleast 10 rows are their not less than that\n"
                "• COMPARATIVE ANALYSIS & METHODOLOGY (800-900 words): Detailed examination of research "
                "  methods, algorithmic approaches, evaluation metrics, and performance comparisons with "
                "  statistical validation and technical assessment\n"
                "• CURRENT WORK & EMERGING TRENDS (1200-1500 words): In-depth analysis of contemporary "
                "  developments, novel methodologies, technological innovations, and empirical findings "
                "  from recent publications with performance evaluation and impact assessment\n"
                "• FUTURE DIRECTIONS & RESEARCH OPPORTUNITIES (600-700 words): Identification of research "
                "  gaps, emerging challenges, technological prospects, and recommended investigation pathways "
                "  with strategic research planning and methodological considerations\n"
                "• CONCLUSION & IMPLICATIONS (400-500 words): Comprehensive synthesis of findings, "
                "  theoretical contributions, practical implications, and scholarly recommendations\n"
                "• REFERENCES (Minimum 20 scholarly citations): Complete bibliographic compilation in "
                "  IEEE/APA format with peer-reviewed sources, conference proceedings, and authoritative "
                "  technical publications\n\n"

                "TECHNICAL REQUIREMENTS:\n"
                "• Implement comprehensive citation strategy with numerical referencing [1-20+]\n"
                "• Maintain consistent academic voice with formal, objective, third-person perspective\n"
                "• Include quantitative performance metrics, statistical comparisons, and empirical validation\n"
                "• Ensure chronological accuracy and methodological precision in all technical descriptions\n"
                "• Provide structured Related Work table with columns: Paper Title | Authors | Year | "
                "  Method/Algorithm | Performance Metrics | Key Contributions | Limitations | Future Work\n"
                "• Integrate conversation history and previous research context seamlessly throughout narrative\n"
                "• Deliver publication-ready manuscript with professional formatting and academic standards"
            ),

            agent=writer,
            context=[research_task,detailed_research_task,rapid_research_task],
            async_execution=False,
        )

    return research_task, write_task


# Enhanced Crew function with memory
def run_crew_for_topic(topic, use_documents=False):
    try:
        research_task, write_task = create_tasks(topic, use_documents)
        researcher, writer = create_agents(use_documents)

        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff(inputs={'topic': topic})

        # Ensure we return the final result properly
        if hasattr(result, 'raw'):
            return result.raw
        elif isinstance(result, dict) and 'final_output' in result:
            return result['final_output']
        else:
            return str(result)

    except Exception as e:
        st.error(f"Error in crew execution: {str(e)}")
        return f"Research completed with some limitations. Topic: {topic}\n\nError encountered: {str(e)}"


# Sidebar with enhanced styling
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

    st.markdown("### 📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF or Word documents",
        type=['pdf', 'docx', 'doc'],
        accept_multiple_files=True,
        help="Upload multiple PDF or Word documents to create a knowledge base"
    )

    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing documents..."):
            documents_data = []

            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                            "application/msword"]:
                    text = extract_text_from_docx(uploaded_file)
                else:
                    continue

                if text.strip():
                    documents_data.append((uploaded_file.name, text))

            if documents_data:
                # Create vector database
                vector_db, processed_docs = create_vector_database(documents_data)
                st.session_state.vector_db = vector_db
                st.session_state.documents = processed_docs

                st.markdown(f'<div class="success-box">✅ Processed {len(documents_data)} documents</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="info-box">📊 Created {len(processed_docs)} text chunks for search</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">❌ No text could be extracted from the uploaded files</div>',
                            unsafe_allow_html=True)

    # Memory Management Section
    st.markdown("### 🧠 Memory Management")
    if st.session_state.memory_context:
        st.markdown(
            f'<div class="sidebar-metric"><span class="metric-number">{len(st.session_state.memory_context)}</span><div class="metric-label">Memories Stored</div></div>',
            unsafe_allow_html=True)

        if st.button("🗑️ Clear Memory", key="clear_memory"):
            st.session_state.memory_context = []
            st.success("Memory cleared!")

        # Show recent memories
        if st.checkbox("📖 Show Recent Memories"):
            st.markdown("**Recent Conversations:**")
            for i, memory in enumerate(st.session_state.memory_context[-3:], 1):
                with st.expander(f"Memory {i} - {memory['timestamp'][:19]}"):
                    st.write(f"**Query:** {memory['query'][:50]}...")
                    st.write(f"**Mode:** {memory['research_mode']}")
    else:
        st.info("No conversation memory yet")

    # Statistics
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        doc_count = len(st.session_state.documents) if st.session_state.documents else 0
        st.markdown(
            f'<div class="sidebar-metric" style="border: 2px solid black; padding: 10px; border-radius: 5px; text-align: center;"><span class="metric-number">{doc_count}</span><div class="metric-label">Documents</div></div>',
            unsafe_allow_html=True)
    with col2:
        chat_count = len(st.session_state.chat_history)
        st.markdown(
            f'<div class="sidebar-metric" style="border: 2px solid black; padding: 10px; border-radius: 5px; text-align: center;"><span class="metric-number">{chat_count}</span><div class="metric-label">Chats</div></div>',
            unsafe_allow_html=True)

    # About Section
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div style="color: black; font-size: 0.9rem; line-height: 1.6;">
    <b>🌐 Internet Research:</b><br>
    • Real-time web search<br>
    • Latest trends and technologies<br>
    • Market analysis<br><br>

    <b>📚 Document Research:</b><br>
    • Upload PDFs and Word documents<br>
    • Create vector embeddings<br>
    • Semantic search through documents<br><br>

    <b>🧠 Memory Features:</b><br>
    • Conversation history tracking<br>
    • Context-aware responses<br>
    • Continuous chat capability<br>
    • Memory-based research continuity
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Clear All Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.vector_db = None
        st.session_state.documents = []
        st.session_state.chat_history = []
        st.session_state.memory_context = []
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.success("All data cleared!")

    st.markdown('</div>', unsafe_allow_html=True)

# Main Content Area
# Research Mode Selection
st.markdown("## 🔍 Research Mode")
research_mode = st.radio(
    "Choose your research source:",
    options=["Internet Research", "Document-based Research"],
    help="Select whether to research from internet sources or uploaded documents",
    horizontal=True
)

# Chat Interface
st.markdown("## 💬 Research Interface")

# Display chat history in an enhanced format
if st.session_state.chat_history:
    st.markdown("### 📚 Previous Conversations")
    for i, chat in enumerate(st.session_state.chat_history[-5:], 1):
        with st.expander(f"🗨️ Conversation {len(st.session_state.chat_history) - 5 + i} - {chat['timestamp'][:19]}"):
            st.markdown(f'<div class="chat-message">', unsafe_allow_html=True)
            st.markdown(f"**🔍 Query:** {chat['query']}")
            st.markdown(f"**⚙️ Mode:** {chat['research_mode']}")
            st.markdown(f"**📝 Response Preview:** {str(chat['response'])[:300]}...")
            st.markdown('</div>', unsafe_allow_html=True)

# Main Research Interface
st.markdown("### 🚀 Start New Research")
col1, col2 = st.columns([4, 1])

with col1:
    topic = st.text_input(
        "Enter a topic for AI Research:",
        placeholder="e.g., AI in Healthcare, Machine Learning in Finance, Quantum Computing trends...",
        help="Enter any topic you want to research. I'll remember our conversation!"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    submit_button = st.button("🚀 Research", type="primary", use_container_width=True)

# Processing and Results
if submit_button and topic:
    if topic.strip():
        use_documents = research_mode == "Document-based Research"

        # Check if documents are available when document mode is selected
        if use_documents and (not st.session_state.vector_db or not st.session_state.documents):
            st.markdown(
                '<div class="warning-box">⚠️ Please upload documents first to use document-based research mode.</div>',
                unsafe_allow_html=True)
        else:
            with st.spinner("🔍 Researching and writing article... This may take a few minutes."):
                try:
                    # Run the crew
                    result = run_crew_for_topic(topic.strip(), use_documents)

                    # Add to memory and chat history
                    add_to_memory(topic.strip(), result, research_mode)

                    chat_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "query": topic.strip(),
                        "response": result,
                        "research_mode": research_mode
                    }
                    st.session_state.chat_history.append(chat_entry)

                    # Keep only last 10 chat entries
                    if len(st.session_state.chat_history) > 10:
                        st.session_state.chat_history = st.session_state.chat_history[-10:]

                    # Success message
                    mode_text = "document-based" if use_documents else "internet-based"
                    st.markdown(
                        f'<div class="success-box">✅ {mode_text.title()} research completed and added to memory!</div>',
                        unsafe_allow_html=True)

                    # Display results
                    st.markdown("## 📋 Research Results")

                    # Create tabs for better organization
                    if use_documents:
                        tab1, tab2, tab3, tab4 = st.tabs(
                            ["📄 Research Article", "🔍 Document Search", "🧠 Memory Explorer", "💾 Downloads"])
                    else:
                        tab1, tab2, tab3 = st.tabs(["📄 Research Article", "🧠 Memory Explorer", "💾 Downloads"])

                    with tab1:
                        if result:
                            st.markdown(f'<div class="research-result">', unsafe_allow_html=True)
                            st.markdown(f"### 📖 {mode_text.title()} Research Report")
                            st.markdown("---")

                            # Format the result better
                            formatted_result = str(result).replace('\n\n', '\n\n---\n\n')
                            st.markdown(formatted_result)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">⚠️ No result generated. Please try again.</div>',
                                        unsafe_allow_html=True)

                    if use_documents:
                        with tab2:
                            st.markdown("### 🔍 Smart Document Search")
                            search_query = st.text_input("Enter a search query:", key="doc_search",
                                                         placeholder="Search through your uploaded documents...")

                            if search_query:
                                search_results = search_documents(
                                    search_query,
                                    st.session_state.vector_db,
                                    st.session_state.documents,
                                    top_k=5
                                )

                                if search_results:
                                    st.markdown("#### 🎯 Top Search Results:")
                                    for i, result_item in enumerate(search_results, 1):
                                        with st.expander(
                                                f"📄 Result {i} - {result_item['source']} (Relevance: {result_item['score']:.1%})"):
                                            st.markdown(result_item['text'])
                                else:
                                    st.info("🔍 No results found for your query.")

                    memory_tab = tab3 if use_documents else tab2
                    with memory_tab:
                        st.markdown("### 🧠 Conversation Memory Explorer")
                        if st.session_state.memory_context:
                            for i, memory in enumerate(st.session_state.memory_context, 1):
                                with st.expander(f"🗂️ Memory {i} - {memory['timestamp'][:19]}"):
                                    st.markdown(f'<div class="memory-entry">', unsafe_allow_html=True)
                                    st.markdown(f"**🔍 Query:** {memory['query']}")
                                    st.markdown(f"**⚙️ Mode:** {memory['research_mode']}")
                                    st.markdown(f"**📝 Response:** {str(memory['response'])[:500]}...")
                                    st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info("🧠 No memory entries yet. Start a conversation to build memory!")

                    download_tab = tab4 if use_documents else tab3
                    with download_tab:
                        st.markdown("### 💾 Download Options")

                        if result:
                            col1, col2, col3 = st.columns(3)

                            # Text download
                            with col1:
                                file_prefix = "document_based" if use_documents else "internet_based"
                                st.download_button(
                                    label="📄 Download as Text",
                                    data=str(result),
                                    file_name=f"{file_prefix}_research_{topic.replace(' ', '_').lower()}_{st.session_state.current_session_id}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )

                            # PDF download
                            with col2:
                                try:
                                    pdf_buffer = create_pdf_report(result, topic, research_mode)
                                    st.download_button(
                                        label="📑 Download as PDF",
                                        data=pdf_buffer.read(),
                                        file_name=f"{file_prefix}_research_{topic.replace(' ', '_').lower()}_{st.session_state.current_session_id}.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                                except Exception as e:
                                    st.error(f"PDF generation error: {str(e)}")

                            # Memory download
                            with col3:
                                if st.session_state.memory_context:
                                    memory_json = json.dumps(st.session_state.memory_context, indent=2)
                                    st.download_button(
                                        label="🧠 Download Memory",
                                        data=memory_json,
                                        file_name=f"conversation_memory_{st.session_state.current_session_id}.json",
                                        mime="application/json",
                                        use_container_width=True
                                    )

                            # Additional download formats
                            st.markdown("---")
                            st.markdown("#### 📊 Additional Formats")

                            col4, col5 = st.columns(2)

                            with col4:
                                # HTML format
                                html_content = f"""
                                <!DOCTYPE html>
                                <html>
                                <head>
                                    <title>{topic} - Research Report</title>
                                    <style>
                                        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                                        h1 {{ color: #2C3E50; }}
                                        h2 {{ color: #667eea; }}
                                        .header {{ border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                                        .content {{ margin-top: 20px; }}
                                        .footer {{ margin-top: 40px; font-size: 0.9em; color: #666; }}
                                    </style>
                                </head>
                                <body>
                                    <div class="header">
                                        <h1>{topic}</h1>
                                        <p><strong>Research Mode:</strong> {research_mode}</p>
                                        <p><strong>Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
                                    </div>
                                    <div class="content">
                                        {str(result).replace(chr(10), '<br>')}
                                    </div>
                                    <div class="footer">
                                        <p>Generated by CrewAI Research Assistant</p>
                                    </div>
                                </body>
                                </html>
                                """

                                st.download_button(
                                    label="🌐 Download as HTML",
                                    data=html_content,
                                    file_name=f"{file_prefix}_research_{topic.replace(' ', '_').lower()}_{st.session_state.current_session_id}.html",
                                    mime="text/html",
                                    use_container_width=True
                                )

                            with col5:
                                # Markdown format
                                markdown_content = f"""# {topic}

**Research Mode:** {research_mode}  
**Generated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}  
**Session ID:** {st.session_state.current_session_id}

---

## Research Report

{str(result)}

---

*Generated by CrewAI Research Assistant*
"""

                                st.download_button(
                                    label="📝 Download as Markdown",
                                    data=markdown_content,
                                    file_name=f"{file_prefix}_research_{topic.replace(' ', '_').lower()}_{st.session_state.current_session_id}.md",
                                    mime="text/markdown",
                                    use_container_width=True
                                )
                        else:
                            st.info("📭 No content available for download.")

                except Exception as e:
                    st.markdown(f'<div class="warning-box">❌ An error occurred: {str(e)}</div>', unsafe_allow_html=True)
                    st.info("Please check your API keys and try again.")
    else:
        st.markdown('<div class="warning-box">⚠️ Please enter a valid topic before submitting.</div>',
                    unsafe_allow_html=True)

elif submit_button and not topic:
    st.markdown('<div class="warning-box">⚠️ Please enter a topic for research.</div>', unsafe_allow_html=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 15px; margin-top: 2rem;">
    <h4 style="color: #2C3E50; margin-bottom: 1rem;">🤖 Academic Research Assistant 🤖</h4>
    <p style="color: #7f8c8d; margin-bottom: 1rem;">
        Enhanced with Advanced Memory • Powered by CrewAI, OpenAI, Streamlit, and SentenceTransformers
    </p>
    <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
        <span style="background: rgba(102, 126, 234, 0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #2C3E50; font-weight: 500;">🌐 Internet Research</span>
        <span style="background: rgba(118, 75, 162, 0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #2C3E50; font-weight: 500;">📚 Document Analysis</span>
        <span style="background: rgba(102, 126, 234, 0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #2C3E50; font-weight: 500;">🧠 Memory System</span>
        <span style="background: rgba(118, 75, 162, 0.2); padding: 0.5rem 1rem; border-radius: 20px; color: #2C3E50; font-weight: 500;">📑 PDF Export</span>
    </div>
</div>
""", unsafe_allow_html=True)
