"""
Advanced PDF Parser for Research Papers

This module provides comprehensive parsing capabilities for research papers,
extracting structured information including:
- Title, authors, abstract, sections
- Tables and figures
- References
- Metadata
"""

import fitz  # PyMuPDF
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from logger import get_logger
from exceptions import PDFParsingError

logger = get_logger(__name__)


@dataclass
class Section:
    """Represents a section in the research paper"""
    title: str
    content: str
    level: int
    page_number: int
    subsections: List['Section'] = field(default_factory=list)


@dataclass
class Figure:
    """Represents a figure in the research paper"""
    caption: str
    page_number: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class Table:
    """Represents a table in the research paper"""
    caption: str
    content: str
    page_number: int


@dataclass
class ResearchPaper:
    """Structured representation of a research paper"""
    title: str
    authors: List[str]
    abstract: str
    sections: List[Section]
    references: List[str]
    figures: List[Figure]
    tables: List[Table]
    metadata: Dict[str, str]
    full_text: str


class ResearchPaperParser:
    """
    Advanced parser for research papers with structure-aware extraction
    """
    
    def __init__(self):
        self.section_patterns = [
            r'^(\d+\.?\s+)?abstract\s*$',
            r'^(\d+\.?\s+)?introduction\s*$',
            r'^(\d+\.?\s+)?related\s+work\s*$',
            r'^(\d+\.?\s+)?methodology\s*$',
            r'^(\d+\.?\s+)?method(s)?\s*$',
            r'^(\d+\.?\s+)?approach\s*$',
            r'^(\d+\.?\s+)?experiment(s)?\s*$',
            r'^(\d+\.?\s+)?result(s)?\s*$',
            r'^(\d+\.?\s+)?discussion\s*$',
            r'^(\d+\.?\s+)?conclusion(s)?\s*$',
            r'^(\d+\.?\s+)?reference(s)?\s*$',
            r'^(\d+\.?\s+)?acknowledgment(s)?\s*$',
        ]
    
    def parse(self, pdf_path: str) -> ResearchPaper:
        """
        Parse a research paper PDF and extract structured information
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ResearchPaper object with extracted information
        """
        logger.info(f"Parsing PDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        
        # Extract metadata
        metadata = self._extract_metadata(doc)
        
        # Extract full text with structure
        full_text, text_blocks = self._extract_text_with_structure(doc)
        
        # Extract title and authors (usually on first page)
        title, authors = self._extract_title_and_authors(doc, text_blocks)
        
        # Extract abstract
        abstract = self._extract_abstract(text_blocks)
        
        # Extract sections
        sections = self._extract_sections(text_blocks)
        
        # Extract references
        references = self._extract_references(text_blocks, doc)
        
        # Extract figures and tables
        figures = self._extract_figures(doc)
        tables = self._extract_tables(doc)
        
        doc.close()
        
        return ResearchPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            references=references,
            figures=figures,
            tables=tables,
            metadata=metadata,
            full_text=full_text
        )
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, str]:
        """Extract PDF metadata"""
        metadata = doc.metadata
        return {
            'author': metadata.get('author', ''),
            'title': metadata.get('title', ''),
            'subject': metadata.get('subject', ''),
            'keywords': metadata.get('keywords', ''),
            'creator': metadata.get('creator', ''),
            'producer': metadata.get('producer', ''),
            'creation_date': metadata.get('creationDate', ''),
            'page_count': str(doc.page_count)
        }
    
    def _extract_text_with_structure(self, doc: fitz.Document) -> Tuple[str, List[Dict]]:
        """
        Extract text while preserving structure information
        
        Returns:
            Tuple of (full_text, text_blocks)
            text_blocks contains: page_num, text, font_size, bbox, etc.
        """
        full_text = []
        text_blocks = []
        
        for page_num, page in enumerate(doc):
            # Get text blocks with formatting information
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = ""
                        font_size = 0
                        
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                            font_size = max(font_size, span.get("size", 0))
                        
                        if line_text.strip():
                            text_blocks.append({
                                'page_num': page_num,
                                'text': line_text.strip(),
                                'font_size': font_size,
                                'bbox': block.get('bbox'),
                            })
                            full_text.append(line_text.strip())
        
        return "\n".join(full_text), text_blocks
    
    def _extract_title_and_authors(self, doc: fitz.Document, text_blocks: List[Dict]) -> Tuple[str, List[str]]:
        """
        Extract title and authors from the first page
        Title is usually the largest font on the first page
        """
        if not text_blocks:
            return "Unknown", []
        
        # Get blocks from first page
        first_page_blocks = [b for b in text_blocks if b['page_num'] == 0]
        
        if not first_page_blocks:
            return "Unknown", []
        
        # Find the largest font size (likely the title)
        max_font_size = max(b['font_size'] for b in first_page_blocks)
        
        # Title is usually the text with the largest font
        title_candidates = [b['text'] for b in first_page_blocks 
                          if b['font_size'] >= max_font_size * 0.95]
        
        title = " ".join(title_candidates[:3]) if title_candidates else "Unknown"
        
        # Authors are typically right after title with smaller font
        authors = []
        author_font_size = max_font_size * 0.6
        
        for block in first_page_blocks[:15]:  # Check first 15 blocks
            if author_font_size * 0.8 <= block['font_size'] <= author_font_size * 1.2:
                # Heuristic: authors often contain commas or "and"
                text = block['text']
                if ',' in text or ' and ' in text.lower() or '@' in text:
                    # Split by common separators
                    potential_authors = re.split(r',|\s+and\s+', text)
                    authors.extend([a.strip() for a in potential_authors if a.strip()])
        
        return title, authors[:10]  # Limit to 10 authors
    
    def _extract_abstract(self, text_blocks: List[Dict]) -> str:
        """Extract abstract section"""
        abstract_lines = []
        in_abstract = False
        
        for i, block in enumerate(text_blocks[:50]):  # Abstract usually in first 50 blocks
            text = block['text'].lower()
            
            if re.match(r'^abstract\s*$', text):
                in_abstract = True
                continue
            
            if in_abstract:
                # Stop at next section
                if self._is_section_header(block['text']):
                    break
                abstract_lines.append(block['text'])
        
        return " ".join(abstract_lines).strip()
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text is a section header"""
        text_lower = text.lower().strip()
        
        for pattern in self.section_patterns:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True
        
        # Also check for numbered sections like "1. Introduction"
        if re.match(r'^\d+\.?\s+[A-Z][a-z]+', text):
            return True
            
        return False
    
    def _extract_sections(self, text_blocks: List[Dict]) -> List[Section]:
        """Extract sections with hierarchy"""
        sections = []
        current_section = None
        current_content = []
        
        for block in text_blocks:
            text = block['text']
            
            if self._is_section_header(text):
                # Save previous section
                if current_section:
                    current_section.content = " ".join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                current_section = Section(
                    title=text,
                    content="",
                    level=1,  # Could be enhanced to detect subsection levels
                    page_number=block['page_num']
                )
                current_content = []
            elif current_section:
                current_content.append(text)
        
        # Save last section
        if current_section:
            current_section.content = " ".join(current_content).strip()
            sections.append(current_section)
        
        return sections
    
    def _extract_references(self, text_blocks: List[Dict], doc: fitz.Document) -> List[str]:
        """Extract references/bibliography"""
        references = []
        in_references = False
        current_ref = []
        
        for block in text_blocks:
            text = block['text']
            text_lower = text.lower().strip()
            
            # Detect references section
            if re.match(r'^(references?|bibliography)\s*$', text_lower):
                in_references = True
                continue
            
            if in_references:
                # References typically start with [1], [2] or 1., 2.
                if re.match(r'^\[\d+\]|\d+\.', text):
                    if current_ref:
                        references.append(" ".join(current_ref).strip())
                    current_ref = [text]
                else:
                    current_ref.append(text)
        
        if current_ref:
            references.append(" ".join(current_ref).strip())
        
        return references
    
    def _extract_figures(self, doc: fitz.Document) -> List[Figure]:
        """Extract figure captions and locations"""
        figures = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Find figure captions (e.g., "Figure 1:", "Fig. 1:")
            pattern = r'(Figure|Fig\.?)\s+\d+[:\.]?\s*([^\n]+)'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                caption = match.group(0)
                
                # Try to find the bounding box
                areas = page.search_for(caption)
                bbox = areas[0] if areas else (0, 0, 0, 0)
                
                figures.append(Figure(
                    caption=caption,
                    page_number=page_num,
                    bbox=bbox
                ))
        
        return figures
    
    def _extract_tables(self, doc: fitz.Document) -> List[Table]:
        """Extract table captions and content"""
        tables = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Find table captions
            pattern = r'(Table)\s+\d+[:\.]?\s*([^\n]+)'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                caption = match.group(0)
                
                # Extract surrounding text as table content (simplified)
                # In a more advanced implementation, you could use table detection
                tables.append(Table(
                    caption=caption,
                    content="",  # Could be enhanced with actual table extraction
                    page_number=page_num
                ))
        
        return tables


def parse_research_paper(pdf_path: str) -> ResearchPaper:
    """
    Convenience function to parse a research paper
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        ResearchPaper object
    """
    parser = ResearchPaperParser()
    return parser.parse(pdf_path)
