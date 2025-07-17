"""
Real-Time Bias & Hallucination Report Generator.

This module detects gender/religion/caste bias using rule-based + NLI models
and flags hallucinated legal citations using a fact verifier module.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Import our real LLM client
try:
    from src.utils.llm_client import get_legal_llm, LLMResponse
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

logger = logging.getLogger(__name__)

class BiasType(Enum):
    """Types of bias that can be detected."""
    GENDER = "gender"
    RELIGION = "religion"
    CASTE = "caste"
    RACE = "race"
    ECONOMIC = "economic"
    REGIONAL = "regional"

class SeverityLevel(Enum):
    """Severity levels for bias detection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class BiasInstance:
    """A detected instance of bias."""
    bias_type: BiasType
    severity: SeverityLevel
    text_snippet: str
    explanation: str
    confidence: float
    start_pos: int
    end_pos: int
    suggested_alternative: Optional[str] = None

@dataclass
class HallucinationInstance:
    """A detected instance of hallucination."""
    citation: str
    text_snippet: str
    reason: str
    confidence: float
    start_pos: int
    end_pos: int
    verified_status: str  # "not_found", "incorrect", "suspicious"

@dataclass
class BiasReport:
    """Complete bias and hallucination report."""
    bias_instances: List[BiasInstance]
    hallucination_instances: List[HallucinationInstance]
    overall_bias_score: float
    overall_hallucination_score: float
    summary: str
    recommendations: List[str]

class BiasDetector:
    """
    Real-time bias and hallucination detector for legal text.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the bias detector."""
        self.config = self._load_config(config_path)

        # Load bias detection patterns
        self.bias_patterns = self._load_bias_patterns()

        # Load legal citation patterns
        self.citation_patterns = self._load_citation_patterns()

        # Load known legal database (simplified)
        self.legal_database = self._load_legal_database()

        logger.info("Bias Detector initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration for bias detection."""
        default_config = {
            'bias_threshold': 0.6,
            'hallucination_threshold': 0.7,
            'enable_gender_bias': True,
            'enable_religion_bias': True,
            'enable_caste_bias': True,
            'enable_citation_check': True,
            'severity_weights': {
                'low': 0.25,
                'medium': 0.5,
                'high': 0.75,
                'critical': 1.0
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def _load_bias_patterns(self) -> Dict[BiasType, Dict]:
        """Load bias detection patterns."""
        return {
            BiasType.GENDER: {
                'biased_terms': [
                    'hysterical woman', 'emotional female', 'irrational woman',
                    'typical woman', 'woman driver', 'housewife mentality',
                    'male breadwinner', 'man of the house', 'boys will be boys'
                ],
                'gendered_assumptions': [
                    r'(?:she|her)\s+(?:must have|obviously|clearly)\s+(?:provoked|asked for)',
                    r'(?:he|his)\s+(?:natural|inherent)\s+(?:aggression|dominance)',
                    r'women\s+(?:are|tend to be)\s+(?:more|less)\s+(?:emotional|rational|capable)'
                ],
                'pronouns': ['he', 'she', 'his', 'her', 'him'],
                'severity_keywords': {
                    'critical': ['inherently', 'naturally', 'biologically'],
                    'high': ['obviously', 'clearly', 'typical'],
                    'medium': ['tends to', 'usually', 'often'],
                    'low': ['might', 'could', 'sometimes']
                }
            },
            BiasType.RELIGION: {
                'biased_terms': [
                    'religious fanatic', 'fundamentalist mindset', 'backward community',
                    'minority appeasement', 'communal tendency', 'religious extremist'
                ],
                'religious_stereotypes': [
                    r'(?:muslim|hindu|christian|sikh)\s+(?:community|people)\s+(?:are|tend to be)\s+(?:violent|peaceful|aggressive)',
                    r'(?:typical|characteristic)\s+(?:muslim|hindu|christian|sikh)\s+(?:behavior|mentality)',
                    r'(?:all|most)\s+(?:muslims|hindus|christians|sikhs)\s+(?:believe|practice|follow)'
                ],
                'severity_keywords': {
                    'critical': ['all', 'every', 'inherently'],
                    'high': ['typical', 'characteristic', 'naturally'],
                    'medium': ['most', 'generally', 'usually'],
                    'low': ['some', 'few', 'occasionally']
                }
            },
            BiasType.CASTE: {
                'biased_terms': [
                    'upper caste mentality', 'lower caste behavior', 'caste-based character',
                    'scheduled caste tendency', 'brahminical attitude', 'dalit mindset'
                ],
                'caste_stereotypes': [
                    r'(?:upper|lower)\s+caste\s+(?:people|community)\s+(?:are|tend to be)',
                    r'(?:brahmin|kshatriya|vaishya|shudra|dalit)\s+(?:nature|character|mentality)',
                    r'caste\s+(?:determines|influences|affects)\s+(?:behavior|character|ability)'
                ],
                'severity_keywords': {
                    'critical': ['determines', 'inherent', 'born with'],
                    'high': ['naturally', 'typically', 'characteristically'],
                    'medium': ['generally', 'often', 'usually'],
                    'low': ['sometimes', 'might', 'could']
                }
            }
        }

    def _load_citation_patterns(self) -> Dict:
        """Load legal citation patterns for hallucination detection."""
        return {
            'case_citations': [
                r'([A-Z][a-zA-Z\s&]+)\s+v\.?\s+([A-Z][a-zA-Z\s&]+)\s*(?:\(\d{4}\)|\d{4})',
                r'([A-Z][a-zA-Z\s&]+)\s+vs\.?\s+([A-Z][a-zA-Z\s&]+)\s*(?:\(\d{4}\)|\d{4})',
                r'([A-Z][a-zA-Z\s&]+)\s+versus\s+([A-Z][a-zA-Z\s&]+)\s*(?:\(\d{4}\)|\d{4})'
            ],
            'section_citations': [
                r'Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?([A-Z][a-zA-Z\s]+(?:Act|Code))',
                r'Sec\.?\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?([A-Z][a-zA-Z\s]+(?:Act|Code))',
                r'Article\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?Constitution'
            ],
            'court_citations': [
                r'(Supreme Court|High Court|District Court|Sessions Court)\s+(?:of\s+)?([A-Z][a-zA-Z\s]+)?',
                r'(SC|HC|DC)\s+(?:of\s+)?([A-Z][a-zA-Z\s]+)?'
            ]
        }

    def _load_legal_database(self) -> Dict:
        """Load simplified legal database for fact checking."""
        # This would typically be loaded from a comprehensive legal database
        return {
            'known_cases': {
                'Kesavananda Bharati v. State of Kerala': {'year': 1973, 'court': 'Supreme Court'},
                'Maneka Gandhi v. Union of India': {'year': 1978, 'court': 'Supreme Court'},
                'Vishaka v. State of Rajasthan': {'year': 1997, 'court': 'Supreme Court'},
                'Indra Sawhney v. Union of India': {'year': 1992, 'court': 'Supreme Court'}
            },
            'known_sections': {
                'Section 302 IPC': 'Murder',
                'Section 307 IPC': 'Attempt to murder',
                'Section 420 IPC': 'Cheating',
                'Section 498A IPC': 'Cruelty by husband or relatives',
                'Article 14': 'Right to equality',
                'Article 19': 'Right to freedom',
                'Article 21': 'Right to life and personal liberty'
            },
            'known_acts': [
                'Indian Penal Code', 'Criminal Procedure Code', 'Indian Evidence Act',
                'Constitution of India', 'Indian Contract Act', 'Transfer of Property Act'
            ]
        }

    def analyze_text(self, text: str) -> BiasReport:
        """
        Analyze text for bias and hallucinations using LLM + rule-based methods.

        Args:
            text: Legal text to analyze

        Returns:
            BiasReport: Complete analysis report
        """
        logger.info("Starting enhanced bias and hallucination analysis")

        if HAS_LLM:
            return self._analyze_with_llm(text)
        else:
            logger.warning("LLM not available, using rule-based analysis only")
            return self._analyze_rule_based(text)

    def _analyze_with_llm(self, text: str) -> BiasReport:
        """Analyze using LLM + rule-based methods."""
        try:
            llm_client = get_legal_llm()

            # Get LLM bias analysis
            response = llm_client.detect_bias(text)

            if response.success:
                # Parse LLM response and combine with rule-based
                llm_results = self._parse_llm_bias_response(response.content)
                rule_based_results = self._analyze_rule_based(text)

                # Combine results
                return self._combine_bias_results(llm_results, rule_based_results, text)
            else:
                logger.error(f"LLM bias analysis failed: {response.error}")
                return self._analyze_rule_based(text)

        except Exception as e:
            logger.error(f"Error in LLM bias analysis: {e}")
            return self._analyze_rule_based(text)

    def _analyze_rule_based(self, text: str) -> BiasReport:
        """Original rule-based analysis."""
        # Detect bias instances
        bias_instances = []
        if self.config['enable_gender_bias']:
            bias_instances.extend(self._detect_gender_bias(text))
        if self.config['enable_religion_bias']:
            bias_instances.extend(self._detect_religion_bias(text))
        if self.config['enable_caste_bias']:
            bias_instances.extend(self._detect_caste_bias(text))

        # Detect hallucinations
        hallucination_instances = []
        if self.config['enable_citation_check']:
            hallucination_instances.extend(self._detect_citation_hallucinations(text))

        # Calculate overall scores
        overall_bias_score = self._calculate_bias_score(bias_instances)
        overall_hallucination_score = self._calculate_hallucination_score(hallucination_instances)

        # Generate summary and recommendations
        summary = self._generate_summary(bias_instances, hallucination_instances)
        recommendations = self._generate_recommendations(bias_instances, hallucination_instances)

        return BiasReport(
            bias_instances=bias_instances,
            hallucination_instances=hallucination_instances,
            overall_bias_score=overall_bias_score,
            overall_hallucination_score=overall_hallucination_score,
            summary=summary,
            recommendations=recommendations
        )

    def _parse_llm_bias_response(self, llm_content: str) -> Dict:
        """Parse LLM bias detection response."""
        try:
            # Try to extract JSON
            if '{' in llm_content and '}' in llm_content:
                json_start = llm_content.find('{')
                json_end = llm_content.rfind('}') + 1
                json_str = llm_content[json_start:json_end]
                return json.loads(json_str)
            else:
                # Parse structured text
                return self._parse_bias_text(llm_content)
        except Exception as e:
            logger.error(f"Error parsing LLM bias response: {e}")
            return {}

    def _parse_bias_text(self, text: str) -> Dict:
        """Parse structured bias analysis text."""
        result = {
            'bias_instances': [],
            'hallucination_instances': [],
            'overall_score': 0.0
        }

        # Extract bias instances
        bias_pattern = r'(?:bias|biased)[^\n]*([^\n]+)'
        bias_matches = re.findall(bias_pattern, text, re.IGNORECASE)

        for match in bias_matches:
            result['bias_instances'].append({
                'text': match.strip(),
                'type': 'general',
                'severity': 'medium'
            })

        # Extract overall score
        score_match = re.search(r'(?:score|rating)[:\s]*([0-9.]+)', text, re.IGNORECASE)
        if score_match:
            result['overall_score'] = float(score_match.group(1))

        return result

    def _combine_bias_results(self, llm_results: Dict, rule_results: BiasReport, text: str) -> BiasReport:
        """Combine LLM and rule-based bias detection results."""
        # Start with rule-based results
        combined_bias = list(rule_results.bias_instances)
        combined_hallucinations = list(rule_results.hallucination_instances)

        # Add LLM-detected bias instances
        for llm_bias in llm_results.get('bias_instances', []):
            # Convert LLM format to our format
            bias_instance = BiasInstance(
                bias_type=BiasType.GENDER,  # Default, could be enhanced
                text_snippet=llm_bias.get('text', ''),
                severity=SeverityLevel(llm_bias.get('severity', 'medium')),
                confidence=0.8,  # LLM confidence
                start_pos=0,
                end_pos=len(llm_bias.get('text', '')),
                suggested_alternative=llm_bias.get('suggestion', ''),
                context="LLM-detected bias"
            )
            combined_bias.append(bias_instance)

        # Recalculate scores
        overall_bias_score = max(
            rule_results.overall_bias_score,
            llm_results.get('overall_score', 0.0)
        )

        # Enhanced summary
        summary = f"Enhanced analysis (LLM + Rules): {len(combined_bias)} bias instances detected. " + rule_results.summary

        return BiasReport(
            bias_instances=combined_bias,
            hallucination_instances=combined_hallucinations,
            overall_bias_score=overall_bias_score,
            overall_hallucination_score=rule_results.overall_hallucination_score,
            summary=summary,
            recommendations=rule_results.recommendations + ["Consider LLM-suggested improvements"]
        )

    def _detect_gender_bias(self, text: str) -> List[BiasInstance]:
        """Detect gender bias in text."""
        bias_instances = []
        patterns = self.bias_patterns[BiasType.GENDER]

        # Check for biased terms
        for term in patterns['biased_terms']:
            for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                severity = self._determine_severity(term, patterns['severity_keywords'])
                confidence = 0.8 if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else 0.6

                bias_instances.append(BiasInstance(
                    bias_type=BiasType.GENDER,
                    severity=severity,
                    text_snippet=match.group(),
                    explanation=f"Potentially biased gender-related term: '{term}'",
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggested_alternative=self._suggest_gender_neutral_alternative(term)
                ))

        # Check for gendered assumptions
        for pattern in patterns['gendered_assumptions']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                bias_instances.append(BiasInstance(
                    bias_type=BiasType.GENDER,
                    severity=SeverityLevel.HIGH,
                    text_snippet=match.group(),
                    explanation="Gendered assumption or stereotype detected",
                    confidence=0.7,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggested_alternative="Consider using gender-neutral language"
                ))

        return bias_instances

    def _detect_religion_bias(self, text: str) -> List[BiasInstance]:
        """Detect religious bias in text."""
        bias_instances = []
        patterns = self.bias_patterns[BiasType.RELIGION]

        # Check for biased terms
        for term in patterns['biased_terms']:
            for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                severity = self._determine_severity(term, patterns['severity_keywords'])
                confidence = 0.8 if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else 0.6

                bias_instances.append(BiasInstance(
                    bias_type=BiasType.RELIGION,
                    severity=severity,
                    text_snippet=match.group(),
                    explanation=f"Potentially biased religious term: '{term}'",
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggested_alternative="Use neutral, respectful language"
                ))

        # Check for religious stereotypes
        for pattern in patterns['religious_stereotypes']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                bias_instances.append(BiasInstance(
                    bias_type=BiasType.RELIGION,
                    severity=SeverityLevel.HIGH,
                    text_snippet=match.group(),
                    explanation="Religious stereotype or generalization detected",
                    confidence=0.75,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggested_alternative="Avoid generalizations about religious communities"
                ))

        return bias_instances

    def _detect_caste_bias(self, text: str) -> List[BiasInstance]:
        """Detect caste-based bias in text."""
        bias_instances = []
        patterns = self.bias_patterns[BiasType.CASTE]

        # Check for biased terms
        for term in patterns['biased_terms']:
            for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                severity = self._determine_severity(term, patterns['severity_keywords'])
                confidence = 0.9 if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL] else 0.7

                bias_instances.append(BiasInstance(
                    bias_type=BiasType.CASTE,
                    severity=severity,
                    text_snippet=match.group(),
                    explanation=f"Potentially biased caste-related term: '{term}'",
                    confidence=confidence,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggested_alternative="Use respectful, non-discriminatory language"
                ))

        # Check for caste stereotypes
        for pattern in patterns['caste_stereotypes']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                bias_instances.append(BiasInstance(
                    bias_type=BiasType.CASTE,
                    severity=SeverityLevel.CRITICAL,
                    text_snippet=match.group(),
                    explanation="Caste-based stereotype or discrimination detected",
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    suggested_alternative="Avoid caste-based characterizations"
                ))

        return bias_instances

    def _detect_citation_hallucinations(self, text: str) -> List[HallucinationInstance]:
        """Detect hallucinated or incorrect legal citations."""
        hallucination_instances = []

        # Check case citations
        for pattern in self.citation_patterns['case_citations']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                case_name = f"{match.group(1)} v. {match.group(2)}"
                verification_result = self._verify_case_citation(case_name)

                if verification_result['status'] != 'verified':
                    hallucination_instances.append(HallucinationInstance(
                        citation=case_name,
                        text_snippet=match.group(),
                        reason=verification_result['reason'],
                        confidence=verification_result['confidence'],
                        start_pos=match.start(),
                        end_pos=match.end(),
                        verified_status=verification_result['status']
                    ))

        # Check section citations
        for pattern in self.citation_patterns['section_citations']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                section_ref = f"Section {match.group(1)}"
                if len(match.groups()) > 1:
                    section_ref += f" {match.group(2)}"

                verification_result = self._verify_section_citation(section_ref)

                if verification_result['status'] != 'verified':
                    hallucination_instances.append(HallucinationInstance(
                        citation=section_ref,
                        text_snippet=match.group(),
                        reason=verification_result['reason'],
                        confidence=verification_result['confidence'],
                        start_pos=match.start(),
                        end_pos=match.end(),
                        verified_status=verification_result['status']
                    ))

        return hallucination_instances

    def _verify_case_citation(self, case_name: str) -> Dict:
        """Verify if a case citation exists in the legal database."""
        # Normalize case name for comparison
        normalized_name = re.sub(r'\s+', ' ', case_name.strip())

        # Check against known cases
        for known_case in self.legal_database['known_cases']:
            if self._case_names_similar(normalized_name, known_case):
                return {'status': 'verified', 'reason': 'Found in database', 'confidence': 0.9}

        # If not found, mark as suspicious
        return {
            'status': 'not_found',
            'reason': 'Case not found in legal database',
            'confidence': 0.7
        }

    def _verify_section_citation(self, section_ref: str) -> Dict:
        """Verify if a section citation is correct."""
        # Check against known sections
        for known_section in self.legal_database['known_sections']:
            if section_ref.lower() in known_section.lower():
                return {'status': 'verified', 'reason': 'Found in database', 'confidence': 0.9}

        # Check if the act is known
        for act in self.legal_database['known_acts']:
            if act.lower() in section_ref.lower():
                return {'status': 'verified', 'reason': 'Act is known', 'confidence': 0.6}

        return {
            'status': 'suspicious',
            'reason': 'Section or Act not found in database',
            'confidence': 0.5
        }

    def _case_names_similar(self, name1: str, name2: str) -> bool:
        """Check if two case names are similar."""
        # Simple similarity check (can be enhanced with fuzzy matching)
        name1_words = set(name1.lower().split())
        name2_words = set(name2.lower().split())

        # Remove common words
        common_words = {'v', 'vs', 'versus', 'and', 'the', 'of', 'in', 'state', 'union'}
        name1_words -= common_words
        name2_words -= common_words

        if not name1_words or not name2_words:
            return False

        # Calculate Jaccard similarity
        intersection = len(name1_words & name2_words)
        union = len(name1_words | name2_words)

        return (intersection / union) > 0.6 if union > 0 else False

    def _determine_severity(self, term: str, severity_keywords: Dict) -> SeverityLevel:
        """Determine severity level based on keywords."""
        term_lower = term.lower()

        for severity, keywords in severity_keywords.items():
            if any(keyword in term_lower for keyword in keywords):
                return SeverityLevel(severity)

        return SeverityLevel.MEDIUM  # Default severity

    def _suggest_gender_neutral_alternative(self, biased_term: str) -> str:
        """Suggest gender-neutral alternatives."""
        alternatives = {
            'hysterical woman': 'emotional person',
            'irrational woman': 'person acting irrationally',
            'typical woman': 'person',
            'male breadwinner': 'primary earner',
            'man of the house': 'head of household',
            'boys will be boys': 'inappropriate behavior should be addressed'
        }

        return alternatives.get(biased_term.lower(), "Use gender-neutral language")

    def _calculate_bias_score(self, bias_instances: List[BiasInstance]) -> float:
        """Calculate overall bias score."""
        if not bias_instances:
            return 0.0

        total_score = 0.0
        for instance in bias_instances:
            severity_weight = self.config['severity_weights'][instance.severity.value]
            total_score += instance.confidence * severity_weight

        # Normalize by number of instances and scale to 0-1
        return min(total_score / len(bias_instances), 1.0)

    def _calculate_hallucination_score(self, hallucination_instances: List[HallucinationInstance]) -> float:
        """Calculate overall hallucination score."""
        if not hallucination_instances:
            return 0.0

        total_score = sum(instance.confidence for instance in hallucination_instances)
        return min(total_score / len(hallucination_instances), 1.0)

    def _generate_summary(self, bias_instances: List[BiasInstance],
                         hallucination_instances: List[HallucinationInstance]) -> str:
        """Generate summary of analysis."""
        summary_parts = []

        if bias_instances:
            bias_count = len(bias_instances)
            bias_types = set(instance.bias_type.value for instance in bias_instances)
            summary_parts.append(f"Found {bias_count} potential bias instances across {len(bias_types)} categories: {', '.join(bias_types)}")
        else:
            summary_parts.append("No significant bias detected")

        if hallucination_instances:
            hall_count = len(hallucination_instances)
            summary_parts.append(f"Found {hall_count} potentially incorrect or unverified citations")
        else:
            summary_parts.append("All citations appear to be valid")

        return ". ".join(summary_parts) + "."

    def _generate_recommendations(self, bias_instances: List[BiasInstance],
                                hallucination_instances: List[HallucinationInstance]) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []

        if bias_instances:
            bias_types = set(instance.bias_type.value for instance in bias_instances)

            if BiasType.GENDER.value in bias_types:
                recommendations.append("Review text for gender-neutral language and avoid gender stereotypes")

            if BiasType.RELIGION.value in bias_types:
                recommendations.append("Ensure respectful language when referring to religious communities")

            if BiasType.CASTE.value in bias_types:
                recommendations.append("Remove caste-based characterizations and use inclusive language")

            # Add severity-based recommendations
            high_severity = [i for i in bias_instances if i.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]]
            if high_severity:
                recommendations.append("Address high-severity bias instances immediately")

        if hallucination_instances:
            recommendations.append("Verify all legal citations against authoritative sources")
            recommendations.append("Double-check case names, section numbers, and act references")

        if not bias_instances and not hallucination_instances:
            recommendations.append("Text appears to be free of significant bias and citation errors")

        return recommendations

def main():
    """Example usage of Bias Detector."""
    # Sample text with potential bias and hallucinations
    sample_text = """
    The hysterical woman clearly provoked the incident, as is typical woman behavior.
    This case is similar to Fake Case v. Non-existent Party (2025) where the Supreme Court ruled.
    Under Section 999 of the Imaginary Act, the accused should be punished.
    The Muslim community tends to be more aggressive in such matters.
    """

    # Initialize detector
    detector = BiasDetector()

    # Analyze text
    report = detector.analyze_text(sample_text)

    # Display results
    print("=" * 60)
    print("BIAS & HALLUCINATION ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nOverall Bias Score: {report.overall_bias_score:.2f}")
    print(f"Overall Hallucination Score: {report.overall_hallucination_score:.2f}")

    print(f"\nSummary: {report.summary}")

    if report.bias_instances:
        print(f"\nBIAS INSTANCES ({len(report.bias_instances)}):")
        for i, instance in enumerate(report.bias_instances, 1):
            print(f"{i}. {instance.bias_type.value.upper()} ({instance.severity.value})")
            print(f"   Text: '{instance.text_snippet}'")
            print(f"   Explanation: {instance.explanation}")
            print(f"   Confidence: {instance.confidence:.2f}")
            if instance.suggested_alternative:
                print(f"   Suggestion: {instance.suggested_alternative}")
            print()

    if report.hallucination_instances:
        print(f"HALLUCINATION INSTANCES ({len(report.hallucination_instances)}):")
        for i, instance in enumerate(report.hallucination_instances, 1):
            print(f"{i}. Citation: '{instance.citation}'")
            print(f"   Status: {instance.verified_status}")
            print(f"   Reason: {instance.reason}")
            print(f"   Confidence: {instance.confidence:.2f}")
            print()

    if report.recommendations:
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
