# src/utils/legal_analyzer.py
"""Legal document analysis and risk detection utilities"""

import re
from typing import List, Dict, Any, Tuple
from ..ai.gemini_client import generate_text_with_gemini
from ..ai.prompts import get_legal_prompt

class LegalAnalyzer:
    """Analyzes legal documents for risks and simplification"""
    
    def __init__(self):
        self.risk_keywords = {
            'high': [
                'penalty', 'fine', 'forfeit', 'liquidated damages', 'automatic renewal',
                'non-refundable', 'irrevocable', 'indemnify', 'hold harmless',
                'unlimited liability', 'personal guarantee', 'compound interest'
            ],
            'medium': [
                'may', 'discretion', 'reasonable', 'material breach', 'cure period',
                'notice', 'consent required', 'approval', 'modification'
            ]
        }
    
    async def analyze_document(self, document_text: str, language: str = 'en') -> Dict[str, Any]:
        """Perform comprehensive legal document analysis using full context"""
        
        print(f"📄 Analyzing complete document ({len(document_text)} chars) with full context...")
        
        # Get simplified version using FULL document context
        print("🔄 Simplifying entire document...")
        simplified_text = await self.simplify_document(document_text, language)
        
        # Detect risks using FULL document context
        print("⚠️ Analyzing risks across entire document...")
        risks = await self.detect_risks(document_text, language)
        
        # Extract key clauses
        print("🔍 Extracting key clauses...")
        key_clauses = self.extract_key_clauses(document_text)
        
        print(f"✅ Analysis complete: {len(simplified_text)} chars simplified, {len(risks)} risks found")
        
        return {
            'original_text': document_text,
            'simplified_text': simplified_text,
            'risks': risks,
            'key_clauses': key_clauses,
            'risk_summary': self.summarize_risks(risks),
            'language': language,
            'analysis_method': 'full_context'
        }
    
    async def simplify_document(self, document_text: str, language: str = 'en') -> str:
        """Simplify legal document into plain language with comprehensive rewrite"""
        prompt = get_legal_prompt('simplify')
        
        if language != 'en':
            prompt += f"\n\nRespond in {self.get_language_name(language)}."
        
        # Enhanced prompt for complete document rewrite using full context
        user_prompt = f"""Please explain this legal document in simple, everyday language that anyone can understand.

Document to explain:
{document_text[:8000]}{'...' if len(document_text) > 8000 else ''}

Please provide:
1. What this document is about
2. Key points in simple terms
3. Important dates, fees, or penalties
4. What the person signing needs to know

Use simple language and avoid legal jargon:"""
        
        print(f"📝 Sending simplification request ({len(user_prompt)} chars)...")
        simplified = await generate_text_with_gemini(user_prompt, prompt)
        print(f"✅ Simplification response: {len(simplified)} chars")
        return simplified
    
    async def detect_risks(self, document_text: str, language: str = 'en') -> List[Dict[str, Any]]:
        """Detect and categorize risks in legal document with detailed analysis"""
        prompt = get_legal_prompt('risks')
        
        if language != 'en':
            prompt += f"\n\nRespond in {self.get_language_name(language)}."
        
        # Enhanced prompt for comprehensive risk detection using full document context
        user_prompt = f"""Analyze this legal document for potential risks and problems.

Document to analyze:
{document_text[:8000]}{'...' if len(document_text) > 8000 else ''}

Look for:
- Hidden fees or penalties
- Automatic renewals
- Unfair terms
- Liability issues
- Cancellation problems

For each risk, explain:
- What the risk is
- Why it's concerning
- What the person should know

Use simple language:"""
        
        print(f"⚠️ Sending risk analysis request ({len(user_prompt)} chars)...")
        risk_analysis = await generate_text_with_gemini(user_prompt, prompt)
        print(f"✅ Risk analysis response: {len(risk_analysis)} chars")
        
        # Parse the risk analysis into structured format
        risks = self.parse_risk_analysis(risk_analysis, document_text)
        return risks
    
    def extract_key_clauses(self, document_text: str) -> List[Dict[str, Any]]:
        """Extract important clauses from legal document"""
        clauses = []
        
        # Common legal section patterns
        section_patterns = [
            r'(?i)(termination|cancellation).*?(?=\n\n|\n[A-Z]|\Z)',
            r'(?i)(payment|fee|cost).*?(?=\n\n|\n[A-Z]|\Z)',
            r'(?i)(liability|damages|indemnif).*?(?=\n\n|\n[A-Z]|\Z)',
            r'(?i)(renewal|extension).*?(?=\n\n|\n[A-Z]|\Z)',
            r'(?i)(breach|default|violation).*?(?=\n\n|\n[A-Z]|\Z)'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, document_text, re.DOTALL)
            for match in matches:
                if len(match) > 50:  # Only include substantial clauses
                    clause_type = self.classify_clause(match)
                    clauses.append({
                        'text': match.strip(),
                        'type': clause_type,
                        'risk_level': self.assess_clause_risk(match)
                    })
        
        return clauses[:10]  # Limit to top 10 clauses
    
    def parse_risk_analysis(self, risk_analysis: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse AI risk analysis into structured format"""
        risks = []
        
        # Look for risk indicators in the analysis
        risk_patterns = [
            (r'🔴.*?(?=🔴|🟡|🟢|\Z)', 'high'),
            (r'🟡.*?(?=🔴|🟡|🟢|\Z)', 'medium'),
            (r'🟢.*?(?=🔴|🟡|🟢|\Z)', 'low')
        ]
        
        for pattern, level in risk_patterns:
            matches = re.findall(pattern, risk_analysis, re.DOTALL)
            for match in matches:
                if len(match.strip()) > 20:
                    risks.append({
                        'level': level,
                        'description': match.strip(),
                        'category': self.categorize_risk(match)
                    })
        
        # If no structured risks found, create basic risk assessment
        if not risks:
            basic_risks = self.basic_risk_detection(original_text)
            risks.extend(basic_risks)
        
        return risks
    
    def basic_risk_detection(self, text: str) -> List[Dict[str, Any]]:
        """Basic keyword-based risk detection"""
        risks = []
        text_lower = text.lower()
        
        for level, keywords in self.risk_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Find the sentence containing the keyword
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            risks.append({
                                'level': level,
                                'description': f"Contains '{keyword}': {sentence.strip()[:200]}...",
                                'category': 'keyword_detection'
                            })
                            break
        
        return risks[:5]  # Limit to top 5 risks
    
    def classify_clause(self, clause_text: str) -> str:
        """Classify the type of legal clause"""
        clause_lower = clause_text.lower()
        
        if any(word in clause_lower for word in ['payment', 'fee', 'cost', 'price']):
            return 'financial'
        elif any(word in clause_lower for word in ['termination', 'cancellation', 'end']):
            return 'termination'
        elif any(word in clause_lower for word in ['liability', 'damages', 'responsible']):
            return 'liability'
        elif any(word in clause_lower for word in ['renewal', 'extension', 'continue']):
            return 'renewal'
        else:
            return 'general'
    
    def assess_clause_risk(self, clause_text: str) -> str:
        """Assess risk level of a clause"""
        clause_lower = clause_text.lower()
        
        high_risk_indicators = ['non-refundable', 'penalty', 'forfeit', 'unlimited', 'irrevocable']
        medium_risk_indicators = ['may', 'discretion', 'reasonable', 'material']
        
        if any(indicator in clause_lower for indicator in high_risk_indicators):
            return 'high'
        elif any(indicator in clause_lower for indicator in medium_risk_indicators):
            return 'medium'
        else:
            return 'low'
    
    def categorize_risk(self, risk_text: str) -> str:
        """Categorize the type of risk"""
        risk_lower = risk_text.lower()
        
        if any(word in risk_lower for word in ['fee', 'cost', 'payment', 'money']):
            return 'financial'
        elif any(word in risk_lower for word in ['liability', 'responsible', 'damages']):
            return 'liability'
        elif any(word in risk_lower for word in ['termination', 'cancellation']):
            return 'termination'
        elif any(word in risk_lower for word in ['renewal', 'automatic']):
            return 'renewal'
        else:
            return 'general'
    
    def summarize_risks(self, risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of identified risks"""
        risk_counts = {'high': 0, 'medium': 0, 'low': 0}
        categories = {}
        
        for risk in risks:
            level = risk.get('level', 'low')
            category = risk.get('category', 'general')
            
            risk_counts[level] += 1
            categories[category] = categories.get(category, 0) + 1
        
        overall_risk = 'low'
        if risk_counts['high'] > 0:
            overall_risk = 'high'
        elif risk_counts['medium'] > 2:
            overall_risk = 'medium'
        
        return {
            'overall_risk': overall_risk,
            'risk_counts': risk_counts,
            'categories': categories,
            'total_risks': len(risks)
        }
    
    def get_language_name(self, language_code: str) -> str:
        """Convert language code to full name"""
        languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'hi': 'Hindi',
            'zh': 'Chinese'
        }
        return languages.get(language_code, 'English')

# Global analyzer instance
legal_analyzer = LegalAnalyzer()