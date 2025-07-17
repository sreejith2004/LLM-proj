"""
IPC/CrPC Section Recommender from Case Facts.

This module predicts applicable legal sections (e.g., IPC 302, CrPC 197)
based on case facts and provides explanations.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class LegalCode(Enum):
    """Types of legal codes."""
    IPC = "Indian Penal Code"
    CRPC = "Criminal Procedure Code"
    IEA = "Indian Evidence Act"
    CONSTITUTION = "Constitution of India"

class SectionType(Enum):
    """Types of legal sections."""
    SUBSTANTIVE = "substantive"  # Defines crimes/rights
    PROCEDURAL = "procedural"    # Defines procedures
    EVIDENTIARY = "evidentiary"  # Defines evidence rules
    CONSTITUTIONAL = "constitutional"  # Constitutional provisions

@dataclass
class LegalSection:
    """A legal section with metadata."""
    section_number: str
    code: LegalCode
    title: str
    description: str
    section_type: SectionType
    keywords: List[str]
    related_sections: List[str]
    punishment_details: Optional[str] = None
    examples: List[str] = None

@dataclass
class SectionPrediction:
    """A predicted legal section."""
    section: LegalSection
    confidence: float
    reasoning: str
    fact_matches: List[str]
    keyword_matches: List[str]

@dataclass
class SectionRecommendation:
    """Complete section recommendation."""
    primary_sections: List[SectionPrediction]
    secondary_sections: List[SectionPrediction]
    procedural_sections: List[SectionPrediction]
    overall_analysis: str
    case_category: str
    severity_assessment: str

class LegalSectionDatabase:
    """Database of legal sections with their metadata."""

    def __init__(self):
        """Initialize the legal section database."""
        self.sections = self._load_sections()
        self.keyword_index = self._build_keyword_index()

    def _load_sections(self) -> Dict[str, LegalSection]:
        """Load legal sections database."""
        sections = {}

        # IPC Sections
        ipc_sections = {
            "302": LegalSection(
                section_number="302",
                code=LegalCode.IPC,
                title="Murder",
                description="Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
                section_type=SectionType.SUBSTANTIVE,
                keywords=["murder", "killing", "death", "homicide", "intentional killing", "premeditated"],
                related_sections=["300", "301", "303", "304"],
                punishment_details="Death or life imprisonment with fine",
                examples=["Intentional killing with premeditation", "Killing with knowledge that act is likely to cause death"]
            ),
            "307": LegalSection(
                section_number="307",
                code=LegalCode.IPC,
                title="Attempt to murder",
                description="Whoever does any act with such intention or knowledge, and under such circumstances that, if he by that act caused death, he would be guilty of murder, shall be punished with imprisonment of either description for a term which may extend to ten years, and shall also be liable to fine; and if hurt is caused to any person by such act, the offender shall be liable either to imprisonment for life, or to such punishment as is hereinbefore mentioned.",
                section_type=SectionType.SUBSTANTIVE,
                keywords=["attempt", "murder", "trying to kill", "intention to kill", "failed murder"],
                related_sections=["302", "308", "511"],
                punishment_details="Up to 10 years imprisonment with fine; life imprisonment if hurt is caused",
                examples=["Shooting at someone with intent to kill but missing", "Poisoning attempt that fails"]
            ),
            "420": LegalSection(
                section_number="420",
                code=LegalCode.IPC,
                title="Cheating and dishonestly inducing delivery of property",
                description="Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
                section_type=SectionType.SUBSTANTIVE,
                keywords=["cheating", "fraud", "deception", "dishonest", "property", "inducement", "false representation"],
                related_sections=["415", "417", "418", "419"],
                punishment_details="Up to 7 years imprisonment with fine",
                examples=["Online fraud", "Investment scams", "False representation to obtain money"]
            ),
            "498A": LegalSection(
                section_number="498A",
                code=LegalCode.IPC,
                title="Husband or relative of husband subjecting woman to cruelty",
                description="Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished with imprisonment for a term which may extend to three years and shall also be liable to fine.",
                section_type=SectionType.SUBSTANTIVE,
                keywords=["cruelty", "husband", "wife", "domestic violence", "harassment", "dowry", "torture"],
                related_sections=["304B", "406", "506"],
                punishment_details="Up to 3 years imprisonment with fine",
                examples=["Domestic violence by husband", "Harassment for dowry", "Mental torture by in-laws"]
            ),
            "376": LegalSection(
                section_number="376",
                code=LegalCode.IPC,
                title="Punishment for rape",
                description="Whoever commits rape shall be punished with rigorous imprisonment of either description for a term which shall not be less than ten years, but which may extend to imprisonment for life, and shall also be liable to fine.",
                section_type=SectionType.SUBSTANTIVE,
                keywords=["rape", "sexual assault", "sexual violence", "consent", "force"],
                related_sections=["375", "376A", "376B", "376C", "376D"],
                punishment_details="Minimum 10 years to life imprisonment with fine",
                examples=["Sexual assault without consent", "Rape by person in authority"]
            ),
            "506": LegalSection(
                section_number="506",
                code=LegalCode.IPC,
                title="Punishment for criminal intimidation",
                description="Whoever commits the offence of criminal intimidation shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both; and if threat be to cause death or grievous hurt, or to cause the destruction of any property by fire, or to cause an offence punishable with death or imprisonment for life, or with imprisonment for a term which may extend to seven years, or to impute, unchastity to a woman, shall be punished with imprisonment of either description for a term which may extend to seven years, or with fine, or with both.",
                section_type=SectionType.SUBSTANTIVE,
                keywords=["intimidation", "threat", "criminal threat", "menace", "coercion"],
                related_sections=["503", "504", "505"],
                punishment_details="Up to 2 years or fine; up to 7 years for serious threats",
                examples=["Threatening to kill", "Threatening to destroy property", "Blackmail"]
            )
        }

        # CrPC Sections
        crpc_sections = {
            "197": LegalSection(
                section_number="197",
                code=LegalCode.CRPC,
                title="Prosecution of Judges and public servants",
                description="When any person who is or was a Judge or Magistrate or a public servant not removable from his office save by or with the sanction of the Government is accused of any offence alleged to have been committed by him while acting or purporting to act in the discharge of his official duty, no Court shall take cognizance of such offence except with the previous sanction of the Central Government or the State Government.",
                section_type=SectionType.PROCEDURAL,
                keywords=["public servant", "judge", "magistrate", "sanction", "government approval", "official duty"],
                related_sections=["196", "198", "199"],
                examples=["Prosecution of government officer", "Case against judge", "Sanction for public servant prosecution"]
            ),
            "156": LegalSection(
                section_number="156",
                code=LegalCode.CRPC,
                title="Police officer's power to investigate cognizable case",
                description="Any officer in charge of a police station may, without the order of a Magistrate, investigate any cognizable case which a Court having jurisdiction over the local area within the limits of such station would have power to inquire into or try under the provisions of this Code.",
                section_type=SectionType.PROCEDURAL,
                keywords=["police investigation", "cognizable", "police station", "investigation power"],
                related_sections=["154", "155", "157", "173"],
                examples=["Police investigation without magistrate order", "Cognizable offense investigation"]
            ),
            "161": LegalSection(
                section_number="161",
                code=LegalCode.CRPC,
                title="Examination of witnesses by police",
                description="Any police officer making an investigation under this Chapter may examine orally any person supposed to be acquainted with the facts and circumstances of the case.",
                section_type=SectionType.PROCEDURAL,
                keywords=["witness examination", "police questioning", "investigation", "oral examination"],
                related_sections=["160", "162", "163"],
                examples=["Police questioning witnesses", "Recording witness statements"]
            ),
            "41": LegalSection(
                section_number="41",
                code=LegalCode.CRPC,
                title="When police may arrest without warrant",
                description="Any police officer may without an order from a Magistrate and without a warrant, arrest any person who has been concerned in any cognizable offence, or against whom a reasonable complaint has been made, or credible information has been received, or a reasonable suspicion exists, of his having been so concerned.",
                section_type=SectionType.PROCEDURAL,
                keywords=["arrest", "without warrant", "police arrest", "cognizable offense", "reasonable suspicion"],
                related_sections=["42", "43", "46", "50"],
                examples=["Arrest for cognizable offense", "Arrest on reasonable suspicion"]
            )
        }

        # Constitutional Articles
        constitutional_sections = {
            "14": LegalSection(
                section_number="14",
                code=LegalCode.CONSTITUTION,
                title="Right to equality",
                description="The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.",
                section_type=SectionType.CONSTITUTIONAL,
                keywords=["equality", "equal protection", "discrimination", "fundamental right"],
                related_sections=["15", "16", "17", "18"],
                examples=["Discrimination cases", "Equal treatment claims"]
            ),
            "21": LegalSection(
                section_number="21",
                code=LegalCode.CONSTITUTION,
                title="Right to life and personal liberty",
                description="No person shall be deprived of his life or personal liberty except according to procedure established by law.",
                section_type=SectionType.CONSTITUTIONAL,
                keywords=["life", "liberty", "personal freedom", "due process", "fundamental right"],
                related_sections=["20", "22"],
                examples=["Right to life cases", "Personal liberty violations", "Due process claims"]
            )
        }

        # Combine all sections
        sections.update(ipc_sections)
        sections.update(crpc_sections)
        sections.update(constitutional_sections)

        return sections

    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """Build keyword index for fast lookup."""
        keyword_index = defaultdict(list)

        for section_id, section in self.sections.items():
            for keyword in section.keywords:
                keyword_index[keyword.lower()].append(section_id)

        return dict(keyword_index)

    def get_section(self, section_id: str) -> Optional[LegalSection]:
        """Get section by ID."""
        return self.sections.get(section_id)

    def search_by_keywords(self, keywords: List[str]) -> List[str]:
        """Search sections by keywords."""
        matching_sections = set()

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.keyword_index:
                matching_sections.update(self.keyword_index[keyword_lower])

        return list(matching_sections)

class LegalSectionPredictor:
    """
    Predicts applicable legal sections based on case facts.
    """

    def __init__(self):
        """Initialize the section predictor."""
        self.database = LegalSectionDatabase()
        self.fact_patterns = self._load_fact_patterns()

    def _load_fact_patterns(self) -> Dict:
        """Load fact patterns for section prediction."""
        return {
            "violence_patterns": {
                "murder": ["killed", "murdered", "death", "fatal", "died", "deceased", "homicide", "murder case", "murder", "killing"],
                "attempt_murder": ["tried to kill", "attempted", "shot at", "stabbed", "poisoned", "failed to kill", "attempt to murder"],
                "assault": ["hit", "beaten", "attacked", "assaulted", "injured", "hurt"],
                "intimidation": ["threatened", "menaced", "intimidated", "coerced", "blackmailed"]
            },
            "property_patterns": {
                "theft": ["stole", "stolen", "theft", "burglary", "robbery", "taken"],
                "fraud": ["cheated", "deceived", "fraud", "scam", "false representation", "dishonest"],
                "criminal_breach": ["breach of trust", "misappropriated", "embezzled"]
            },
            "sexual_patterns": {
                "rape": ["rape", "sexual assault", "forced", "without consent", "sexual violence"],
                "harassment": ["sexual harassment", "eve teasing", "molestation", "inappropriate touching"]
            },
            "domestic_patterns": {
                "cruelty": ["domestic violence", "cruelty", "harassment", "torture", "dowry", "in-laws"],
                "dowry": ["dowry death", "dowry harassment", "dowry demand"]
            },
            "procedural_patterns": {
                "investigation": ["investigation", "police inquiry", "evidence collection"],
                "arrest": ["arrested", "custody", "detention", "warrant"],
                "trial": ["trial", "court proceedings", "hearing", "judgment"]
            }
        }

    def predict_sections(self, case_facts: str, case_type: Optional[str] = None) -> SectionRecommendation:
        """
        Predict applicable legal sections based on case facts.

        Args:
            case_facts: Description of case facts
            case_type: Optional case type hint

        Returns:
            SectionRecommendation: Complete recommendation with sections
        """
        logger.info("Predicting legal sections from case facts")

        # Extract keywords from facts
        extracted_keywords = self._extract_keywords(case_facts)

        # Identify fact patterns
        identified_patterns = self._identify_patterns(case_facts)

        # Predict sections
        primary_predictions = self._predict_primary_sections(case_facts, extracted_keywords, identified_patterns)
        secondary_predictions = self._predict_secondary_sections(case_facts, primary_predictions)
        procedural_predictions = self._predict_procedural_sections(case_facts, primary_predictions)

        # Analyze case
        case_category = self._categorize_case(identified_patterns)
        severity_assessment = self._assess_severity(primary_predictions)
        overall_analysis = self._generate_analysis(primary_predictions, secondary_predictions, procedural_predictions)

        return SectionRecommendation(
            primary_sections=primary_predictions,
            secondary_sections=secondary_predictions,
            procedural_sections=procedural_predictions,
            overall_analysis=overall_analysis,
            case_category=case_category,
            severity_assessment=severity_assessment
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        # Enhanced keyword extraction with section reference detection
        text_lower = text.lower()

        # Check for explicit section references first
        section_pattern = r'section\s+(\d+[a-z]?)\s+(ipc|crpc)'
        section_matches = re.findall(section_pattern, text_lower)

        legal_keywords = []

        # Add keywords based on section references
        for section_num, code in section_matches:
            if section_num == '302' and code == 'ipc':
                legal_keywords.extend(["murder", "killing", "death", "homicide", "intentional"])
            elif section_num == '307' and code == 'ipc':
                legal_keywords.extend(["attempt", "murder", "trying", "kill"])
            elif section_num == '420' and code == 'ipc':
                legal_keywords.extend(["cheating", "fraud", "deception"])

        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)

        # Filter relevant legal keywords from database
        all_keywords = set()
        for section in self.database.sections.values():
            all_keywords.update(keyword.lower() for keyword in section.keywords)

        for word in words:
            if word in all_keywords:
                legal_keywords.append(word)

        return list(set(legal_keywords))

    def _identify_patterns(self, text: str) -> Dict[str, List[str]]:
        """Identify fact patterns in text."""
        text_lower = text.lower()
        identified = defaultdict(list)

        for category, patterns in self.fact_patterns.items():
            for pattern_type, keywords in patterns.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        identified[category].append(pattern_type)

        # Remove duplicates
        for category in identified:
            identified[category] = list(set(identified[category]))

        return dict(identified)

    def _predict_primary_sections(self, case_facts: str, keywords: List[str],
                                 patterns: Dict[str, List[str]]) -> List[SectionPrediction]:
        """Predict primary applicable sections."""
        predictions = []

        # Violence-related sections
        if "violence_patterns" in patterns:
            if "murder" in patterns["violence_patterns"]:
                section = self.database.get_section("302")
                if section:
                    confidence = self._calculate_confidence(case_facts, section)
                    predictions.append(SectionPrediction(
                        section=section,
                        confidence=confidence,
                        reasoning="Case facts indicate intentional killing/murder",
                        fact_matches=self._find_fact_matches(case_facts, section.keywords),
                        keyword_matches=[k for k in keywords if k in [kw.lower() for kw in section.keywords]]
                    ))

            if "attempt_murder" in patterns["violence_patterns"]:
                section = self.database.get_section("307")
                if section:
                    confidence = self._calculate_confidence(case_facts, section)
                    predictions.append(SectionPrediction(
                        section=section,
                        confidence=confidence,
                        reasoning="Case facts suggest attempted murder",
                        fact_matches=self._find_fact_matches(case_facts, section.keywords),
                        keyword_matches=[k for k in keywords if k in [kw.lower() for kw in section.keywords]]
                    ))

            if "intimidation" in patterns["violence_patterns"]:
                section = self.database.get_section("506")
                if section:
                    confidence = self._calculate_confidence(case_facts, section)
                    predictions.append(SectionPrediction(
                        section=section,
                        confidence=confidence,
                        reasoning="Criminal intimidation detected",
                        fact_matches=self._find_fact_matches(case_facts, section.keywords),
                        keyword_matches=[k for k in keywords if k in [kw.lower() for kw in section.keywords]]
                    ))

        # Property-related sections
        if "property_patterns" in patterns:
            if "fraud" in patterns["property_patterns"]:
                section = self.database.get_section("420")
                if section:
                    confidence = self._calculate_confidence(case_facts, section)
                    predictions.append(SectionPrediction(
                        section=section,
                        confidence=confidence,
                        reasoning="Fraudulent activity and cheating detected",
                        fact_matches=self._find_fact_matches(case_facts, section.keywords),
                        keyword_matches=[k for k in keywords if k in [kw.lower() for kw in section.keywords]]
                    ))

        # Sexual offense sections
        if "sexual_patterns" in patterns:
            if "rape" in patterns["sexual_patterns"]:
                section = self.database.get_section("376")
                if section:
                    confidence = self._calculate_confidence(case_facts, section)
                    predictions.append(SectionPrediction(
                        section=section,
                        confidence=confidence,
                        reasoning="Sexual assault/rape indicated",
                        fact_matches=self._find_fact_matches(case_facts, section.keywords),
                        keyword_matches=[k for k in keywords if k in [kw.lower() for kw in section.keywords]]
                    ))

        # Domestic violence sections
        if "domestic_patterns" in patterns:
            if "cruelty" in patterns["domestic_patterns"]:
                section = self.database.get_section("498A")
                if section:
                    confidence = self._calculate_confidence(case_facts, section)
                    predictions.append(SectionPrediction(
                        section=section,
                        confidence=confidence,
                        reasoning="Domestic cruelty by husband/relatives",
                        fact_matches=self._find_fact_matches(case_facts, section.keywords),
                        keyword_matches=[k for k in keywords if k in [kw.lower() for kw in section.keywords]]
                    ))

        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)
        return predictions[:5]  # Top 5 predictions

    def _predict_secondary_sections(self, case_facts: str,
                                   primary_predictions: List[SectionPrediction]) -> List[SectionPrediction]:
        """Predict secondary/related sections."""
        secondary_predictions = []

        for primary in primary_predictions:
            for related_section_id in primary.section.related_sections:
                related_section = self.database.get_section(related_section_id)
                if related_section:
                    confidence = self._calculate_confidence(case_facts, related_section) * 0.7  # Lower confidence
                    secondary_predictions.append(SectionPrediction(
                        section=related_section,
                        confidence=confidence,
                        reasoning=f"Related to primary section {primary.section.section_number}",
                        fact_matches=self._find_fact_matches(case_facts, related_section.keywords),
                        keyword_matches=[]
                    ))

        # Remove duplicates and sort
        seen_sections = set()
        unique_secondary = []
        for pred in secondary_predictions:
            if pred.section.section_number not in seen_sections:
                unique_secondary.append(pred)
                seen_sections.add(pred.section.section_number)

        unique_secondary.sort(key=lambda x: x.confidence, reverse=True)
        return unique_secondary[:3]  # Top 3 secondary

    def _predict_procedural_sections(self, case_facts: str,
                                    primary_predictions: List[SectionPrediction]) -> List[SectionPrediction]:
        """Predict procedural sections."""
        procedural_predictions = []

        # Check for investigation-related procedures
        if any(word in case_facts.lower() for word in ["investigation", "police", "arrest", "custody"]):
            for section_id in ["156", "161", "41"]:
                section = self.database.get_section(section_id)
                if section:
                    confidence = 0.6  # Standard confidence for procedural
                    procedural_predictions.append(SectionPrediction(
                        section=section,
                        confidence=confidence,
                        reasoning="Procedural section for investigation/arrest",
                        fact_matches=[],
                        keyword_matches=[]
                    ))

        # Check for public servant cases
        if any(word in case_facts.lower() for word in ["public servant", "government", "officer", "judge"]):
            section = self.database.get_section("197")
            if section:
                procedural_predictions.append(SectionPrediction(
                    section=section,
                    confidence=0.7,
                    reasoning="Public servant involved - sanction may be required",
                    fact_matches=[],
                    keyword_matches=[]
                ))

        return procedural_predictions

    def _calculate_confidence(self, case_facts: str, section: LegalSection) -> float:
        """Calculate confidence score for section applicability."""
        case_facts_lower = case_facts.lower()
        import re

        # Check for explicit section reference first
        section_pattern = f"section\\s+{section.section_number}\\s+(ipc|crpc)"
        if re.search(section_pattern, case_facts_lower):
            return 0.9  # High confidence if section is explicitly mentioned

        # Base confidence
        confidence = 0.3

        # Keyword matching
        keyword_matches = 0
        for keyword in section.keywords:
            if keyword.lower() in case_facts_lower:
                keyword_matches += 1
                confidence += 0.1

        # Boost for multiple keyword matches
        if keyword_matches > 2:
            confidence += 0.1

        # Boost for exact phrase matches
        for example in (section.examples or []):
            if example.lower() in case_facts_lower:
                confidence += 0.15

        return min(confidence, 1.0)

    def _find_fact_matches(self, case_facts: str, keywords: List[str]) -> List[str]:
        """Find matching facts in case description."""
        matches = []
        case_facts_lower = case_facts.lower()

        for keyword in keywords:
            if keyword.lower() in case_facts_lower:
                # Find sentence containing the keyword
                sentences = case_facts.split('.')
                for sentence in sentences:
                    if keyword.lower() in sentence.lower():
                        matches.append(sentence.strip())
                        break

        return matches[:3]  # Limit to 3 matches

    def _categorize_case(self, patterns: Dict[str, List[str]]) -> str:
        """Categorize the case based on identified patterns."""
        if "violence_patterns" in patterns:
            if "murder" in patterns["violence_patterns"]:
                return "Homicide Case"
            elif "attempt_murder" in patterns["violence_patterns"]:
                return "Attempted Homicide Case"
            else:
                return "Violence/Assault Case"

        elif "sexual_patterns" in patterns:
            return "Sexual Offense Case"

        elif "domestic_patterns" in patterns:
            return "Domestic Violence Case"

        elif "property_patterns" in patterns:
            return "Property Crime Case"

        else:
            return "General Criminal Case"

    def _assess_severity(self, primary_predictions: List[SectionPrediction]) -> str:
        """Assess case severity based on predicted sections."""
        if not primary_predictions:
            return "Low Severity"

        # Check for serious offenses
        serious_sections = ["302", "376", "307"]  # Murder, rape, attempt to murder

        for prediction in primary_predictions:
            if prediction.section.section_number in serious_sections and prediction.confidence > 0.3:
                return "Critical Severity"

        # Check for medium severity
        medium_sections = ["420", "498A", "506"]
        for prediction in primary_predictions:
            if prediction.section.section_number in medium_sections and prediction.confidence > 0.5:
                return "Medium Severity"

        return "Low Severity"

    def _generate_analysis(self, primary: List[SectionPrediction],
                          secondary: List[SectionPrediction],
                          procedural: List[SectionPrediction]) -> str:
        """Generate overall analysis."""
        analysis_parts = []

        if primary:
            primary_sections = [p.section.section_number for p in primary[:3]]
            analysis_parts.append(f"Primary applicable sections: {', '.join(primary_sections)}")

        if secondary:
            analysis_parts.append(f"Related sections to consider: {len(secondary)} additional sections")

        if procedural:
            analysis_parts.append(f"Procedural requirements: {len(procedural)} procedural sections apply")

        if not analysis_parts:
            return "No clear legal sections identified from the provided facts."

        return ". ".join(analysis_parts) + "."

def main():
    """Example usage of Legal Section Predictor."""
    # Sample case facts
    sample_cases = [
        {
            "title": "Murder Case",
            "facts": "The accused intentionally killed the victim with a knife after a heated argument. The victim died on the spot due to fatal injuries."
        },
        {
            "title": "Fraud Case",
            "facts": "The accused cheated the complainant by making false representations about investment returns and dishonestly induced him to transfer money."
        },
        {
            "title": "Domestic Violence Case",
            "facts": "The husband and his relatives subjected the wife to physical and mental cruelty, demanding additional dowry and threatening her."
        },
        {
            "title": "Attempt Murder Case",
            "facts": "The accused shot at the victim with intent to kill but the victim survived with injuries. The accused had planned the attack."
        }
    ]

    # Initialize predictor
    predictor = LegalSectionPredictor()

    print("=" * 70)
    print("LEGAL SECTION PREDICTOR DEMO")
    print("=" * 70)

    for i, case in enumerate(sample_cases, 1):
        print(f"\n{i}. {case['title']}")
        print("-" * 50)
        print(f"Facts: {case['facts']}")

        # Predict sections
        recommendation = predictor.predict_sections(case['facts'])

        print(f"\nCase Category: {recommendation.case_category}")
        print(f"Severity: {recommendation.severity_assessment}")
        print(f"Analysis: {recommendation.overall_analysis}")

        if recommendation.primary_sections:
            print(f"\nPRIMARY SECTIONS ({len(recommendation.primary_sections)}):")
            for j, pred in enumerate(recommendation.primary_sections, 1):
                section = pred.section
                print(f"{j}. Section {section.section_number} {section.code.value}")
                print(f"   Title: {section.title}")
                print(f"   Confidence: {pred.confidence:.2f}")
                print(f"   Reasoning: {pred.reasoning}")
                if section.punishment_details:
                    print(f"   Punishment: {section.punishment_details}")
                print()

        if recommendation.procedural_sections:
            print(f"PROCEDURAL SECTIONS ({len(recommendation.procedural_sections)}):")
            for pred in recommendation.procedural_sections:
                section = pred.section
                print(f"- Section {section.section_number} {section.code.value}: {section.title}")

if __name__ == "__main__":
    main()
