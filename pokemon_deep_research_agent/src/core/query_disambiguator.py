"""
Query Disambiguator for Pokemon Deep Research Agent

Implements dynamic query disambiguation and interactive specification elicitation.
Follows ChatGPT's deep research pattern of asking clarifying questions before execution.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .specification_manager import (SpecificationField, SpecificationManager,
                                    TaskSpecification)


@dataclass
class DisambiguationQuestion:
    """Represents a clarifying question for the user."""

    field_name: str
    question: str
    field_type: str
    options: List[str] = None
    required: bool = True
    default_value: Any = None


class QueryDisambiguator:
    """
    Handles query disambiguation and interactive specification elicitation.

    Features:
    - Dynamic question generation based on missing fields
    - Interactive confirmation workflow
    - Context-aware question prioritization
    - User-friendly question formatting
    """

    def __init__(self, spec_manager: SpecificationManager, llm_client=None):
        self.spec_manager = spec_manager
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)

    async def disambiguate_query_interactive(
        self, spec: TaskSpecification, conversation_context: Dict[str, Any] = None
    ) -> TaskSpecification:
        """
        Interactive disambiguation process with user.

        Args:
            spec: Initial specification from query analysis
            conversation_context: Previous conversation context

        Returns:
            Completed and confirmed specification
        """

        print(f"\n🔍 **QUERY DISAMBIGUATION & SPECIFICATION CONFIRMATION**")
        print("=" * 70)

        # Step 1: Show initial specification
        print(self.spec_manager.format_specification_summary(spec))

        # Step 2: Check for missing required fields
        missing_required = self.spec_manager.get_missing_required_fields(spec)

        if missing_required:
            print(f"\n❓ **I need some additional information to proceed:**")

            for field in missing_required:
                answer = await self._ask_field_question(field, conversation_context)
                if answer is not None:
                    spec.fields[field.name] = answer
                    print(f"   ✅ {field.description}: {answer}")

        # Step 3: Confirm optional fields
        optional_fields = self.spec_manager.get_optional_fields_to_confirm(spec)

        if optional_fields:
            print(f"\n⚙️ **Optional settings (press Enter for defaults):**")

            for field in optional_fields:
                answer = await self._ask_optional_field_question(
                    field, conversation_context
                )
                if answer is not None:
                    spec.fields[field.name] = answer
                    print(f"   ✅ {field.description}: {answer}")
                elif field.default_value is not None:
                    spec.fields[field.name] = field.default_value
                    print(f"   📌 {field.description}: {field.default_value} (default)")

        # Step 4: Final confirmation
        print(f"\n📋 **FINAL SPECIFICATION:**")
        print(self.spec_manager.format_specification_summary(spec))

        while True:
            confirm = (
                input("\n✅ Proceed with this specification? [Y/n/edit]: ")
                .strip()
                .lower()
            )

            if confirm in ["", "y", "yes"]:
                spec.confirmed = True
                print("🚀 **Specification confirmed! Starting deep research...**")
                break
            elif confirm in ["n", "no"]:
                print("❌ **Research cancelled.**")
                return None
            elif confirm in ["edit", "e"]:
                spec = await self._edit_specification_interactive(spec)
                print(f"\n📋 **UPDATED SPECIFICATION:**")
                print(self.spec_manager.format_specification_summary(spec))
            else:
                print("Please enter 'y' (yes), 'n' (no), or 'edit'")

        return spec

    async def _ask_field_question(
        self, field: SpecificationField, context: Dict[str, Any] = None
    ) -> Any:
        """Ask user a question for a required field."""

        question = field.question or f"Please specify {field.description}:"

        if field.field_type == "select" and field.options:
            print(f"\n❓ {question}")
            for i, option in enumerate(field.options, 1):
                print(f"   {i}. {option}")

            while True:
                try:
                    choice = input(f"Choose (1-{len(field.options)}): ").strip()
                    if choice.isdigit():
                        idx = int(choice) - 1
                        if 0 <= idx < len(field.options):
                            return field.options[idx]

                    # Try direct text match
                    choice_lower = choice.lower()
                    for option in field.options:
                        if option.lower() == choice_lower:
                            return option

                    print(
                        f"Please enter a number 1-{len(field.options)} or the option name"
                    )
                except (ValueError, KeyboardInterrupt):
                    print(f"Please enter a number 1-{len(field.options)}")

        elif field.field_type == "boolean":
            print(f"\n❓ {question}")
            while True:
                choice = input("Enter y/n: ").strip().lower()
                if choice in ["y", "yes", "true", "1"]:
                    return True
                elif choice in ["n", "no", "false", "0"]:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")

        elif field.field_type == "pokemon_list":
            print(f"\n❓ {question}")
            print("Enter Pokemon names separated by commas:")

            while True:
                pokemon_input = input("Pokemon: ").strip()
                if pokemon_input:
                    pokemon_list = [p.strip().lower() for p in pokemon_input.split(",")]
                    return pokemon_list
                else:
                    print("Please enter at least one Pokemon name")

        else:  # text field
            print(f"\n❓ {question}")
            while True:
                answer = input("Answer: ").strip()
                if answer:
                    return answer
                else:
                    print("This field is required. Please provide an answer.")

    async def _ask_optional_field_question(
        self, field: SpecificationField, context: Dict[str, Any] = None
    ) -> Optional[Any]:
        """Ask user a question for an optional field."""

        question = field.question or f"Specify {field.description}:"
        default_text = (
            f" (default: {field.default_value})"
            if field.default_value is not None
            else ""
        )

        if field.field_type == "select" and field.options:
            print(f"\n⚙️ {question}{default_text}")
            for i, option in enumerate(field.options, 1):
                default_marker = " ⭐" if option == field.default_value else ""
                print(f"   {i}. {option}{default_marker}")

            choice = input(
                f"Choose (1-{len(field.options)}) or Enter for default: "
            ).strip()

            if not choice:
                return field.default_value

            try:
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(field.options):
                        return field.options[idx]

                # Try direct text match
                choice_lower = choice.lower()
                for option in field.options:
                    if option.lower() == choice_lower:
                        return option

                return field.default_value
            except ValueError:
                return field.default_value

        elif field.field_type == "boolean":
            print(f"\n⚙️ {question}{default_text}")
            choice = input("Enter y/n or Enter for default: ").strip().lower()

            if not choice:
                return field.default_value
            elif choice in ["y", "yes", "true", "1"]:
                return True
            elif choice in ["n", "no", "false", "0"]:
                return False
            else:
                return field.default_value

        else:  # text field
            print(f"\n⚙️ {question}{default_text}")
            answer = input("Answer (or Enter for default): ").strip()
            return answer if answer else field.default_value

    async def _edit_specification_interactive(
        self, spec: TaskSpecification
    ) -> TaskSpecification:
        """Allow user to edit the specification interactively."""

        schema = self.spec_manager.schemas.get(spec.query_type, [])
        field_map = {field.name: field for field in schema}

        print(f"\n✏️ **EDIT SPECIFICATION**")
        print("Available fields to edit:")

        editable_fields = []
        for i, (field_name, value) in enumerate(spec.fields.items(), 1):
            field = field_map.get(field_name)
            if field:
                print(f"   {i}. {field.description}: {value}")
                editable_fields.append((field_name, field))

        while True:
            choice = (
                input(f"\nChoose field to edit (1-{len(editable_fields)}) or 'done': ")
                .strip()
                .lower()
            )

            if choice == "done":
                break

            try:
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(editable_fields):
                        field_name, field = editable_fields[idx]
                        print(f"\nEditing: {field.description}")
                        print(f"Current value: {spec.fields[field_name]}")

                        new_value = await self._ask_field_question(field)
                        if new_value is not None:
                            spec.fields[field_name] = new_value
                            print(f"✅ Updated {field.description}: {new_value}")
                        continue

                print(f"Please enter a number 1-{len(editable_fields)} or 'done'")
            except (ValueError, KeyboardInterrupt):
                print(f"Please enter a number 1-{len(editable_fields)} or 'done'")

        return spec

    async def generate_dynamic_questions(
        self, spec: TaskSpecification, conversation_context: Dict[str, Any] = None
    ) -> List[DisambiguationQuestion]:
        """
        Generate dynamic clarifying questions using LLM.

        This is an advanced feature that uses LLM to generate contextual questions
        based on the query and conversation history.
        """

        if not self.llm_client:
            return self._generate_static_questions(spec)

        missing_fields = self.spec_manager.get_missing_required_fields(spec)
        optional_fields = self.spec_manager.get_optional_fields_to_confirm(spec)

        context_text = ""
        if conversation_context and conversation_context.get("mentioned_pokemon"):
            context_text = f"\nConversation context: Previously discussed {', '.join(conversation_context['mentioned_pokemon'])}"

        question_prompt = f"""
        Generate natural, contextual questions to clarify this Pokemon research query:
        
        Original Query: "{spec.original_query}"
        Query Type: {spec.query_type.value}
        Current Specification: {spec.fields}{context_text}
        
        Missing Required Fields: {[f.name for f in missing_fields]}
        Optional Fields to Confirm: {[f.name for f in optional_fields]}
        
        Generate natural language questions that:
        1. Are specific to Pokemon research
        2. Consider the conversation context
        3. Help clarify the user's intent
        4. Are easy to understand and answer
        
        Format as a JSON list of questions with field names.
        """

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": question_prompt}], temperature=0.3
            )

            # Parse LLM response and create DisambiguationQuestion objects
            # For now, fall back to static questions
            return self._generate_static_questions(spec)

        except Exception as e:
            self.logger.error(f"Error generating dynamic questions: {e}")
            return self._generate_static_questions(spec)

    def _generate_static_questions(
        self, spec: TaskSpecification
    ) -> List[DisambiguationQuestion]:
        """Generate static questions based on specification schema."""

        questions = []

        missing_fields = self.spec_manager.get_missing_required_fields(spec)
        optional_fields = self.spec_manager.get_optional_fields_to_confirm(spec)

        for field in missing_fields:
            questions.append(
                DisambiguationQuestion(
                    field_name=field.name,
                    question=field.question,
                    field_type=field.field_type,
                    options=field.options,
                    required=True,
                )
            )

        for field in optional_fields:
            questions.append(
                DisambiguationQuestion(
                    field_name=field.name,
                    question=field.question,
                    field_type=field.field_type,
                    options=field.options,
                    required=False,
                    default_value=field.default_value,
                )
            )

        return questions

    async def auto_complete_from_context(
        self, spec: TaskSpecification, conversation_context: Dict[str, Any] = None
    ) -> TaskSpecification:
        """
        Automatically complete specification fields from conversation context.

        This reduces the number of questions needed by inferring information
        from previous conversation turns.
        """

        if not conversation_context:
            return spec

        # Auto-complete Pokemon fields from context
        mentioned_pokemon = conversation_context.get("mentioned_pokemon", [])

        if mentioned_pokemon and spec.query_type.value in [
            "competitive_analysis",
            "moveset_analysis",
        ]:
            if "target_pokemon" not in spec.fields:
                # Use the most recently mentioned Pokemon
                spec.fields["target_pokemon"] = mentioned_pokemon[-1]
                self.logger.info(
                    f"Auto-completed target_pokemon from context: {mentioned_pokemon[-1]}"
                )

        elif mentioned_pokemon and spec.query_type.value == "team_building":
            if "core_pokemon" not in spec.fields:
                spec.fields["core_pokemon"] = mentioned_pokemon[-1]
                self.logger.info(
                    f"Auto-completed core_pokemon from context: {mentioned_pokemon[-1]}"
                )

        elif (
            len(mentioned_pokemon) >= 2
            and spec.query_type.value == "pokemon_comparison"
        ):
            if "pokemon_list" not in spec.fields:
                spec.fields["pokemon_list"] = mentioned_pokemon[-2:]  # Last 2 mentioned
                self.logger.info(
                    f"Auto-completed pokemon_list from context: {mentioned_pokemon[-2:]}"
                )

        # Auto-complete format from conversation themes
        themes = conversation_context.get("themes", [])
        if "competitive" in themes and "format" not in spec.fields:
            spec.fields["format"] = "OU"  # Default competitive format
            self.logger.info("Auto-completed format from competitive theme: OU")

        return spec
