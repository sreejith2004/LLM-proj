"""
Chain-of-Thought (CoT) Reasoning Module for Legal Case Outcome Prediction.

This module implements a structured reasoning approach that breaks down
legal case analysis into: LAW → FACT → ARGUMENT → OUTCOME.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re

# Import our real LLM client instead of transformers
try:
    from src.utils.llm_client import get_legal_llm, LLMResponse
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

logger = logging.getLogger(__name__)

class OutcomeLabel(Enum):
    """Legal case outcome labels."""
    GUILTY = "Guilty"
    NOT_GUILTY = "Not Guilty"
    COMPENSATED = "Compensated"
    DISMISSED = "Dismissed"
    PARTIALLY_GRANTED = "Partially Granted"
    REMANDED = "Remanded"
    ACQUITTED = "Acquitted"
    CONVICTED = "Convicted"

@dataclass
class CoTStep:
    """Represents a single step in Chain-of-Thought reasoning."""
    step_name: str
    reasoning: str
    confidence: float
    key_points: List[str]

@dataclass
class CoTResult:
    """Complete Chain-of-Thought reasoning result."""
    law_analysis: CoTStep
    fact_analysis: CoTStep
    argument_analysis: CoTStep
    outcome_prediction: CoTStep
    final_verdict: OutcomeLabel
    overall_confidence: float
    reasoning_chain: str

class ChainOfThoughtReasoner:
    """
    Chain-of-Thought reasoning system for legal case analysis.

    Implements structured reasoning: LAW → FACT → ARGUMENT → OUTCOME
    """

    def __init__(self, model=None, legal_knowledge_base: Optional[Dict] = None):
        """
        Initialize the CoT reasoner.

        Args:
            model: Language model for reasoning (optional)
            legal_knowledge_base: Dictionary of legal knowledge (optional)
        """
        self.model = model
        self.legal_kb = legal_knowledge_base or self._load_default_legal_kb()

        # Initialize confidence thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }

        logger.info("Chain-of-Thought Reasoner initialized")

    def _load_default_legal_kb(self) -> Dict:
        """Load default legal knowledge base."""
        return {
            'ipc_sections': {
                '302': 'Murder',
                '307': 'Attempt to murder',
                '420': 'Cheating and dishonestly inducing delivery of property',
                '498A': 'Husband or relative of husband subjecting woman to cruelty',
                '376': 'Punishment for rape'
            },
            'crpc_sections': {
                '197': 'Prosecution of Judges and public servants',
                '156': 'Police officer\'s power to investigate cognizable case',
                '161': 'Examination of witnesses by police'
            },
            'legal_principles': [
                'Burden of proof lies on the prosecution',
                'Accused is innocent until proven guilty',
                'Evidence must be beyond reasonable doubt',
                'Circumstantial evidence must form complete chain'
            ]
        }

    def analyze_case(self, case_text: str, case_facts: Optional[str] = None) -> CoTResult:
        """
        Perform complete Chain-of-Thought analysis of a legal case using real LLM.

        Args:
            case_text: Full case text or judgment
            case_facts: Specific case facts (optional)

        Returns:
            CoTResult: Complete reasoning analysis
        """
        logger.info("Starting Chain-of-Thought analysis with LLM")

        if HAS_LLM:
            return self._analyze_with_llm(case_text, case_facts)
        else:
            # Fallback to basic analysis if LLM not available
            logger.warning("LLM not available, using fallback analysis")
            return self._analyze_fallback(case_text, case_facts)

    def _analyze_with_llm(self, case_text: str, case_facts: Optional[str] = None) -> CoTResult:
        """Perform analysis using real LLM."""
        try:
            llm_client = get_legal_llm()

            # Get LLM analysis
            response = llm_client.analyze_legal_case(case_text, "chain_of_thought")

            if not response.success:
                logger.error(f"LLM analysis failed: {response.error}")
                return self._create_error_result(response.error or "LLM analysis failed")

            # Parse LLM response
            return self._parse_llm_response(response.content, case_text)

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._create_error_result(str(e))

    def _parse_llm_response(self, llm_content: str, original_text: str) -> CoTResult:
        """Parse LLM response into structured CoTResult."""
        try:
            # Try to extract JSON if present
            if '{' in llm_content and '}' in llm_content:
                json_start = llm_content.find('{')
                json_end = llm_content.rfind('}') + 1
                json_str = llm_content[json_start:json_end]
                data = json.loads(json_str)
            else:
                # Parse structured text response
                data = self._parse_structured_text(llm_content)

            # Extract steps
            law_step = CoTStep(
                step_name="LAW",
                reasoning=data.get('law_analysis', 'Legal analysis completed'),
                confidence=float(data.get('law_confidence', 0.7)),
                key_points=data.get('law_points', [])
            )

            fact_step = CoTStep(
                step_name="FACT",
                reasoning=data.get('fact_analysis', 'Factual analysis completed'),
                confidence=float(data.get('fact_confidence', 0.7)),
                key_points=data.get('fact_points', [])
            )

            argument_step = CoTStep(
                step_name="ARGUMENT",
                reasoning=data.get('argument_analysis', 'Argument analysis completed'),
                confidence=float(data.get('argument_confidence', 0.7)),
                key_points=data.get('argument_points', [])
            )

            outcome_step = CoTStep(
                step_name="OUTCOME",
                reasoning=data.get('outcome_analysis', 'Outcome prediction completed'),
                confidence=float(data.get('outcome_confidence', 0.7)),
                key_points=data.get('outcome_points', [])
            )

            # Determine final verdict
            verdict_str = data.get('final_verdict', 'Not Guilty')
            try:
                final_verdict = OutcomeLabel(verdict_str)
            except ValueError:
                final_verdict = OutcomeLabel.NOT_GUILTY

            overall_confidence = (law_step.confidence + fact_step.confidence +
                                argument_step.confidence + outcome_step.confidence) / 4

            return CoTResult(
                law_analysis=law_step,
                fact_analysis=fact_step,
                argument_analysis=argument_step,
                outcome_prediction=outcome_step,
                final_verdict=final_verdict,
                overall_confidence=overall_confidence,
                reasoning_chain=llm_content[:500] + "..." if len(llm_content) > 500 else llm_content
            )

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_error_result(f"Failed to parse LLM response: {e}")

    def _parse_structured_text(self, text: str) -> Dict[str, Any]:
        """Parse structured text response from LLM."""
        data = {}

        # Extract sections using regex
        sections = {
            'law_analysis': r'1\.\s*LAW Analysis[:\s]*([^\n]*(?:\n(?!\d+\.).*)*)',
            'fact_analysis': r'2\.\s*FACT Analysis[:\s]*([^\n]*(?:\n(?!\d+\.).*)*)',
            'argument_analysis': r'3\.\s*ARGUMENT Analysis[:\s]*([^\n]*(?:\n(?!\d+\.).*)*)',
            'outcome_analysis': r'4\.\s*OUTCOME[^:]*[:\s]*([^\n]*(?:\n(?!\d+\.).*)*)',
        }

        for key, pattern in sections.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                data[key] = match.group(1).strip()

        # Extract verdict
        verdict_match = re.search(r'(?:verdict|outcome)[:\s]*([^\n]*)', text, re.IGNORECASE)
        if verdict_match:
            verdict_text = verdict_match.group(1).lower()
            if 'guilty' in verdict_text and 'not' not in verdict_text:
                data['final_verdict'] = 'Guilty'
            else:
                data['final_verdict'] = 'Not Guilty'

        # Extract confidence if mentioned
        conf_match = re.search(r'confidence[:\s]*([0-9.]+)', text, re.IGNORECASE)
        if conf_match:
            conf_val = float(conf_match.group(1))
            if conf_val > 1.0:  # If percentage, convert to decimal
                conf_val = conf_val / 100
            data['outcome_confidence'] = conf_val

        return data

    def _create_error_result(self, error_msg: str) -> CoTResult:
        """Create error result when analysis fails."""
        error_step = CoTStep(
            step_name="ERROR",
            reasoning=f"Analysis failed: {error_msg}",
            confidence=0.0,
            key_points=[]
        )

        return CoTResult(
            law_analysis=error_step,
            fact_analysis=error_step,
            argument_analysis=error_step,
            outcome_prediction=error_step,
            final_verdict=OutcomeLabel.NOT_GUILTY,
            overall_confidence=0.0,
            reasoning_chain=f"Chain-of-Thought analysis failed: {error_msg}"
        )

    def _analyze_fallback(self, case_text: str, case_facts: Optional[str] = None) -> CoTResult:
        """Fallback analysis when LLM is not available."""
        logger.info("Using fallback Chain-of-Thought analysis")

        # Step 1: LAW Analysis
        law_step = self._analyze_law(case_text)

        # Step 2: FACT Analysis
        fact_step = self._analyze_facts(case_text, case_facts)

        # Step 3: ARGUMENT Analysis
        argument_step = self._analyze_arguments(case_text, law_step, fact_step)

        # Step 4: OUTCOME Prediction
        outcome_step = self._predict_outcome(case_text, law_step, fact_step, argument_step)

        # Generate final result
        result = self._generate_final_result(law_step, fact_step, argument_step, outcome_step)

        logger.info(f"Fallback CoT analysis completed. Final verdict: {result.final_verdict.value}")
        return result

    def _analyze_law(self, case_text: str) -> CoTStep:
        """Analyze applicable laws and legal provisions."""
        logger.debug("Analyzing applicable laws")

        # Extract legal sections mentioned
        ipc_sections = self._extract_ipc_sections(case_text)
        crpc_sections = self._extract_crpc_sections(case_text)

        # Identify legal principles
        principles = self._identify_legal_principles(case_text)

        # Generate reasoning
        reasoning = f"""
        Legal Analysis:
        - IPC Sections identified: {', '.join(ipc_sections) if ipc_sections else 'None explicitly mentioned'}
        - CrPC Sections identified: {', '.join(crpc_sections) if crpc_sections else 'None explicitly mentioned'}
        - Applicable legal principles: {', '.join(principles[:3]) if principles else 'Standard criminal law principles'}

        The case appears to involve {self._categorize_case_type(case_text)} based on the legal provisions and context.
        """

        key_points = ipc_sections + crpc_sections + principles[:2]
        confidence = self._calculate_law_confidence(ipc_sections, crpc_sections, principles)

        return CoTStep(
            step_name="LAW",
            reasoning=reasoning.strip(),
            confidence=confidence,
            key_points=key_points
        )

    def _analyze_facts(self, case_text: str, case_facts: Optional[str] = None) -> CoTStep:
        """Analyze case facts and evidence."""
        logger.debug("Analyzing case facts")

        # Extract key facts
        facts = self._extract_key_facts(case_text, case_facts)

        # Identify evidence types
        evidence_types = self._identify_evidence_types(case_text)

        # Assess fact strength
        fact_strength = self._assess_fact_strength(facts, evidence_types)

        reasoning = f"""
        Factual Analysis:
        - Key facts identified: {len(facts)} major factual elements
        - Evidence types present: {', '.join(evidence_types) if evidence_types else 'Limited evidence details'}
        - Fact pattern strength: {fact_strength}

        The factual matrix suggests {self._interpret_fact_pattern(facts, evidence_types)}.
        """

        key_points = facts[:3] + evidence_types[:2]
        confidence = self._calculate_fact_confidence(facts, evidence_types)

        return CoTStep(
            step_name="FACT",
            reasoning=reasoning.strip(),
            confidence=confidence,
            key_points=key_points
        )

    def _analyze_arguments(self, case_text: str, law_step: CoTStep, fact_step: CoTStep) -> CoTStep:
        """Analyze legal arguments and their strength."""
        logger.debug("Analyzing legal arguments")

        # Extract prosecution arguments
        prosecution_args = self._extract_prosecution_arguments(case_text)

        # Extract defense arguments
        defense_args = self._extract_defense_arguments(case_text)

        # Assess argument strength
        arg_balance = self._assess_argument_balance(prosecution_args, defense_args)

        reasoning = f"""
        Argument Analysis:
        - Prosecution arguments: {len(prosecution_args)} key contentions identified
        - Defense arguments: {len(defense_args)} counter-arguments identified
        - Argument balance: {arg_balance}

        Based on the legal framework from Step 1 and facts from Step 2,
        the argument analysis reveals {self._synthesize_arguments(prosecution_args, defense_args, law_step, fact_step)}.
        """

        key_points = prosecution_args[:2] + defense_args[:2]
        confidence = self._calculate_argument_confidence(prosecution_args, defense_args)

        return CoTStep(
            step_name="ARGUMENT",
            reasoning=reasoning.strip(),
            confidence=confidence,
            key_points=key_points
        )

    def _predict_outcome(self, case_text: str, law_step: CoTStep,
                        fact_step: CoTStep, argument_step: CoTStep) -> CoTStep:
        """Predict case outcome based on previous analysis."""
        logger.debug("Predicting case outcome")

        # Synthesize all previous steps
        combined_confidence = (law_step.confidence + fact_step.confidence + argument_step.confidence) / 3

        # Predict outcome label
        outcome_label = self._determine_outcome_label(case_text, law_step, fact_step, argument_step)

        # Generate outcome reasoning
        reasoning = f"""
        Outcome Prediction:
        - Predicted verdict: {outcome_label.value}
        - Confidence level: {self._confidence_to_text(combined_confidence)}

        Reasoning chain synthesis:
        1. Legal framework establishes the applicable provisions and principles
        2. Factual analysis reveals the strength of evidence and case circumstances
        3. Argument analysis shows the balance between prosecution and defense positions
        4. Conclusion: Based on the totality of circumstances, the likely outcome is {outcome_label.value}

        This prediction is based on the structured analysis of law, facts, and arguments.
        """

        key_points = [
            f"Predicted: {outcome_label.value}",
            f"Confidence: {combined_confidence:.2f}",
            "Based on comprehensive CoT analysis"
        ]

        return CoTStep(
            step_name="OUTCOME",
            reasoning=reasoning.strip(),
            confidence=combined_confidence,
            key_points=key_points
        )

    def _generate_final_result(self, law_step: CoTStep, fact_step: CoTStep,
                              argument_step: CoTStep, outcome_step: CoTStep) -> CoTResult:
        """Generate the final CoT result."""

        # Determine final verdict
        final_verdict = self._extract_verdict_from_outcome(outcome_step)

        # Calculate overall confidence
        overall_confidence = (law_step.confidence + fact_step.confidence +
                            argument_step.confidence + outcome_step.confidence) / 4

        # Generate reasoning chain summary
        reasoning_chain = f"""
        CHAIN-OF-THOUGHT LEGAL REASONING:

        1. LAW → {law_step.reasoning[:100]}...
        2. FACT → {fact_step.reasoning[:100]}...
        3. ARGUMENT → {argument_step.reasoning[:100]}...
        4. OUTCOME → {outcome_step.reasoning[:100]}...

        FINAL VERDICT: {final_verdict.value} (Confidence: {overall_confidence:.2f})
        """

        return CoTResult(
            law_analysis=law_step,
            fact_analysis=fact_step,
            argument_analysis=argument_step,
            outcome_prediction=outcome_step,
            final_verdict=final_verdict,
            overall_confidence=overall_confidence,
            reasoning_chain=reasoning_chain.strip()
        )

    # Helper methods for extraction and analysis

    def _extract_ipc_sections(self, text: str) -> List[str]:
        """Extract IPC sections from text."""
        pattern = r'(?:IPC|Indian Penal Code)?\s*[Ss]ection\s*(\d+[A-Z]?)'
        matches = re.findall(pattern, text)
        return list(set(matches))

    def _extract_crpc_sections(self, text: str) -> List[str]:
        """Extract CrPC sections from text."""
        pattern = r'(?:CrPC|Criminal Procedure Code)?\s*[Ss]ection\s*(\d+[A-Z]?)'
        matches = re.findall(pattern, text)
        return list(set(matches))

    def _identify_legal_principles(self, text: str) -> List[str]:
        """Identify applicable legal principles."""
        principles = []
        text_lower = text.lower()

        if 'burden of proof' in text_lower:
            principles.append('Burden of proof')
        if 'reasonable doubt' in text_lower:
            principles.append('Beyond reasonable doubt')
        if 'circumstantial evidence' in text_lower:
            principles.append('Circumstantial evidence')
        if 'presumption of innocence' in text_lower:
            principles.append('Presumption of innocence')

        return principles

    def _categorize_case_type(self, text: str) -> str:
        """Categorize the type of legal case."""
        text_lower = text.lower()

        if any(word in text_lower for word in ['murder', 'homicide', 'killing']):
            return 'criminal case involving homicide'
        elif any(word in text_lower for word in ['theft', 'robbery', 'burglary']):
            return 'property crime case'
        elif any(word in text_lower for word in ['fraud', 'cheating', 'forgery']):
            return 'financial crime case'
        elif any(word in text_lower for word in ['assault', 'battery', 'violence']):
            return 'violent crime case'
        else:
            return 'general criminal/civil matter'

    def _extract_key_facts(self, case_text: str, case_facts: Optional[str] = None) -> List[str]:
        """Extract key facts from case text."""
        facts = []

        # Use case_facts if provided, otherwise extract from case_text
        text_to_analyze = case_facts if case_facts else case_text

        # Simple fact extraction (can be enhanced with NLP)
        sentences = text_to_analyze.split('.')
        for sentence in sentences[:10]:  # Limit to first 10 sentences
            if len(sentence.strip()) > 20:  # Filter short sentences
                facts.append(sentence.strip())

        return facts[:5]  # Return top 5 facts

    def _identify_evidence_types(self, text: str) -> List[str]:
        """Identify types of evidence mentioned."""
        evidence_types = []
        text_lower = text.lower()

        evidence_keywords = {
            'witness testimony': ['witness', 'testimony', 'deposed'],
            'documentary evidence': ['document', 'record', 'certificate'],
            'physical evidence': ['physical', 'material', 'object'],
            'circumstantial evidence': ['circumstantial', 'indirect'],
            'expert evidence': ['expert', 'forensic', 'medical']
        }

        for evidence_type, keywords in evidence_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                evidence_types.append(evidence_type)

        return evidence_types

    def _assess_fact_strength(self, facts: List[str], evidence_types: List[str]) -> str:
        """Assess the strength of factual foundation."""
        if len(facts) >= 4 and len(evidence_types) >= 2:
            return "Strong factual foundation"
        elif len(facts) >= 2 and len(evidence_types) >= 1:
            return "Moderate factual foundation"
        else:
            return "Limited factual foundation"

    def _interpret_fact_pattern(self, facts: List[str], evidence_types: List[str]) -> str:
        """Interpret the overall fact pattern."""
        if 'witness testimony' in evidence_types and 'documentary evidence' in evidence_types:
            return "a well-supported case with multiple evidence types"
        elif len(evidence_types) >= 2:
            return "a case with diverse evidence sources"
        else:
            return "a case requiring careful evaluation of available evidence"

    def _extract_prosecution_arguments(self, text: str) -> List[str]:
        """Extract prosecution arguments."""
        # Simple extraction - can be enhanced
        prosecution_indicators = ['prosecution', 'state', 'complainant']
        arguments = []

        sentences = text.split('.')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in prosecution_indicators):
                if len(sentence.strip()) > 30:
                    arguments.append(sentence.strip())

        return arguments[:3]

    def _extract_defense_arguments(self, text: str) -> List[str]:
        """Extract defense arguments."""
        # Simple extraction - can be enhanced
        defense_indicators = ['defense', 'defence', 'accused', 'appellant']
        arguments = []

        sentences = text.split('.')
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in defense_indicators):
                if len(sentence.strip()) > 30:
                    arguments.append(sentence.strip())

        return arguments[:3]

    def _assess_argument_balance(self, prosecution_args: List[str], defense_args: List[str]) -> str:
        """Assess the balance between prosecution and defense arguments."""
        p_count = len(prosecution_args)
        d_count = len(defense_args)

        if p_count > d_count * 1.5:
            return "Prosecution-favored"
        elif d_count > p_count * 1.5:
            return "Defense-favored"
        else:
            return "Balanced arguments"

    def _synthesize_arguments(self, prosecution_args: List[str], defense_args: List[str],
                             law_step: CoTStep, fact_step: CoTStep) -> str:
        """Synthesize argument analysis with law and fact steps."""
        return f"a {self._assess_argument_balance(prosecution_args, defense_args).lower()} presentation that must be evaluated against the legal framework and factual foundation"

    def _determine_outcome_label(self, case_text: str, law_step: CoTStep,
                                fact_step: CoTStep, argument_step: CoTStep) -> OutcomeLabel:
        """Determine the most likely outcome label."""
        text_lower = case_text.lower()

        # Simple rule-based prediction (can be enhanced with ML)
        if 'acquitted' in text_lower or 'not guilty' in text_lower:
            return OutcomeLabel.NOT_GUILTY
        elif 'convicted' in text_lower or 'guilty' in text_lower:
            return OutcomeLabel.GUILTY
        elif 'dismissed' in text_lower:
            return OutcomeLabel.DISMISSED
        elif 'compensation' in text_lower or 'damages' in text_lower:
            return OutcomeLabel.COMPENSATED
        elif 'remand' in text_lower:
            return OutcomeLabel.REMANDED
        else:
            # Default prediction based on confidence levels
            avg_confidence = (law_step.confidence + fact_step.confidence + argument_step.confidence) / 3
            if avg_confidence > 0.7:
                return OutcomeLabel.GUILTY
            else:
                return OutcomeLabel.NOT_GUILTY

    def _extract_verdict_from_outcome(self, outcome_step: CoTStep) -> OutcomeLabel:
        """Extract verdict from outcome step."""
        reasoning = outcome_step.reasoning.lower()

        for label in OutcomeLabel:
            if label.value.lower() in reasoning:
                return label

        # Default fallback
        return OutcomeLabel.DISMISSED

    def _calculate_law_confidence(self, ipc_sections: List[str], crpc_sections: List[str],
                                 principles: List[str]) -> float:
        """Calculate confidence for law analysis."""
        base_confidence = 0.5

        # Boost confidence based on identified elements
        if ipc_sections:
            base_confidence += 0.2
        if crpc_sections:
            base_confidence += 0.1
        if principles:
            base_confidence += 0.1 * min(len(principles), 2)

        return min(base_confidence, 1.0)

    def _calculate_fact_confidence(self, facts: List[str], evidence_types: List[str]) -> float:
        """Calculate confidence for fact analysis."""
        base_confidence = 0.4

        # Boost confidence based on facts and evidence
        base_confidence += 0.1 * min(len(facts), 5)
        base_confidence += 0.1 * min(len(evidence_types), 3)

        return min(base_confidence, 1.0)

    def _calculate_argument_confidence(self, prosecution_args: List[str], defense_args: List[str]) -> float:
        """Calculate confidence for argument analysis."""
        base_confidence = 0.5

        # Boost confidence based on argument availability
        if prosecution_args:
            base_confidence += 0.15
        if defense_args:
            base_confidence += 0.15

        # Boost for balanced arguments
        if abs(len(prosecution_args) - len(defense_args)) <= 1:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _confidence_to_text(self, confidence: float) -> str:
        """Convert confidence score to text."""
        if confidence >= self.confidence_thresholds['high']:
            return "High"
        elif confidence >= self.confidence_thresholds['medium']:
            return "Medium"
        else:
            return "Low"

def main():
    """Example usage of Chain-of-Thought reasoner."""
    # Sample case text
    sample_case = """
    The appellant was charged under Section 302 IPC for the murder of the deceased.
    The prosecution argued that there was sufficient circumstantial evidence to prove guilt beyond reasonable doubt.
    The defense contended that the evidence was insufficient and the accused should be given benefit of doubt.
    The trial court convicted the accused, but the High Court acquitted him on appeal.
    """

    # Initialize reasoner
    reasoner = ChainOfThoughtReasoner()

    # Perform analysis
    result = reasoner.analyze_case(sample_case)

    # Display results
    print("=" * 60)
    print("CHAIN-OF-THOUGHT LEGAL REASONING ANALYSIS")
    print("=" * 60)

    print(f"\n1. LAW ANALYSIS (Confidence: {result.law_analysis.confidence:.2f})")
    print(result.law_analysis.reasoning)

    print(f"\n2. FACT ANALYSIS (Confidence: {result.fact_analysis.confidence:.2f})")
    print(result.fact_analysis.reasoning)

    print(f"\n3. ARGUMENT ANALYSIS (Confidence: {result.argument_analysis.confidence:.2f})")
    print(result.argument_analysis.reasoning)

    print(f"\n4. OUTCOME PREDICTION (Confidence: {result.outcome_prediction.confidence:.2f})")
    print(result.outcome_prediction.reasoning)

    print(f"\nFINAL VERDICT: {result.final_verdict.value}")
    print(f"OVERALL CONFIDENCE: {result.overall_confidence:.2f}")

if __name__ == "__main__":
    main()
