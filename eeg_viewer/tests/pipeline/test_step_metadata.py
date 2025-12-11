"""
Tests for step metadata and phase grouping.

Verifies that all steps have complete metadata for the improved UI.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.pages.pipeline.constants.step_info import (
    STEP_INFO, PHASES, Phase, get_steps_for_phase, get_step_by_script
)
from app.pages.pipeline.steps import STEPS


class TestStepInfoCompleteness:
    """Verify all step info is complete."""
    
    def test_all_8_steps_defined(self):
        """Should have info for all 8 steps."""
        assert len(STEP_INFO) == 8
    
    def test_step_numbers_1_to_8(self):
        """Step numbers should be 1-8."""
        assert set(STEP_INFO.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}
    
    def test_all_steps_have_display_name(self):
        """Each step should have a display name."""
        for num, info in STEP_INFO.items():
            assert info.display_name, f"Step {num} missing display_name"
            assert len(info.display_name) > 3, f"Step {num} display_name too short"
    
    def test_all_steps_have_descriptions(self):
        """Each step should have descriptions."""
        for num, info in STEP_INFO.items():
            assert info.short_description, f"Step {num} missing short_description"
            assert info.detailed_description, f"Step {num} missing detailed_description"
    
    def test_all_steps_have_script_name(self):
        """Each step should have a script name."""
        for num, info in STEP_INFO.items():
            assert info.script_name, f"Step {num} missing script_name"
            assert info.script_name.endswith('.py'), f"Step {num} script_name not a .py file"
    
    def test_all_steps_have_phase(self):
        """Each step should be assigned to a phase."""
        for num, info in STEP_INFO.items():
            assert info.phase in Phase, f"Step {num} has invalid phase"
    
    def test_all_steps_have_time_estimates(self):
        """Each step should have time estimates."""
        for num, info in STEP_INFO.items():
            assert info.time_estimate, f"Step {num} missing time_estimate"
            assert info.time_with_limit, f"Step {num} missing time_with_limit"
    
    def test_all_steps_have_output_info(self):
        """Each step should have output information."""
        for num, info in STEP_INFO.items():
            assert info.output_description, f"Step {num} missing output_description"
            assert info.output_example, f"Step {num} missing output_example"
    
    def test_all_steps_have_button_info(self):
        """Each step should have button text and icon."""
        for num, info in STEP_INFO.items():
            assert info.button_text, f"Step {num} missing button_text"
            assert info.button_icon, f"Step {num} missing button_icon"
    
    def test_all_steps_have_input_structure(self):
        """Each step should have input structure documentation."""
        for num, info in STEP_INFO.items():
            assert info.input_structure, f"Step {num} missing input_structure"
            assert len(info.input_structure) > 50, f"Step {num} input_structure too short"
    
    def test_all_steps_have_output_structure(self):
        """Each step should have output structure documentation."""
        for num, info in STEP_INFO.items():
            assert info.output_structure, f"Step {num} missing output_structure"
            assert len(info.output_structure) > 50, f"Step {num} output_structure too short"


class TestPhaseDefinitions:
    """Verify phase definitions are complete."""
    
    def test_all_3_phases_defined(self):
        """Should have 3 phases."""
        assert len(PHASES) == 3
    
    def test_phases_have_all_fields(self):
        """Each phase should have all required fields."""
        for phase, info in PHASES.items():
            assert info.title, f"Phase {phase} missing title"
            assert info.subtitle, f"Phase {phase} missing subtitle"
            assert info.icon, f"Phase {phase} missing icon"
            assert info.color, f"Phase {phase} missing color"
    
    def test_phase_colors_are_valid_hex(self):
        """Phase colors should be valid hex."""
        for phase, info in PHASES.items():
            assert info.color.startswith('#'), f"Phase {phase} color not hex"
            assert len(info.color) in [4, 7], f"Phase {phase} color invalid length"


class TestPhaseGrouping:
    """Test step grouping by phase."""
    
    def test_processing_phase_has_steps_1_2(self):
        """Processing phase should have steps 1-2."""
        steps = get_steps_for_phase(Phase.PROCESSING)
        step_nums = [s.step_number for s in steps]
        assert 1 in step_nums
        assert 2 in step_nums
    
    def test_connectivity_phase_has_steps_3_6(self):
        """Connectivity phase should have steps 3-6."""
        steps = get_steps_for_phase(Phase.CONNECTIVITY)
        step_nums = [s.step_number for s in steps]
        assert 3 in step_nums
        assert 4 in step_nums
        assert 5 in step_nums
        assert 6 in step_nums
    
    def test_advanced_phase_has_steps_7_8(self):
        """Advanced phase should have steps 7-8."""
        steps = get_steps_for_phase(Phase.ADVANCED)
        step_nums = [s.step_number for s in steps]
        assert 7 in step_nums
        assert 8 in step_nums
    
    def test_all_steps_in_exactly_one_phase(self):
        """Each step should be in exactly one phase."""
        all_step_nums = set()
        for phase in Phase:
            steps = get_steps_for_phase(phase)
            nums = {s.step_number for s in steps}
            # No overlap
            assert len(all_step_nums & nums) == 0, "Step in multiple phases"
            all_step_nums.update(nums)
        
        # All steps accounted for
        assert all_step_nums == {1, 2, 3, 4, 5, 6, 7, 8}


class TestStepInfoLookup:
    """Test step info lookup functions."""
    
    def test_get_step_by_script_fwd(self):
        """Should find fwd.py step."""
        info = get_step_by_script('fwd.py')
        assert info is not None
        assert info.step_number == 1
    
    def test_get_step_by_script_clustering(self):
        """Should find clustering.py step."""
        info = get_step_by_script('clustering.py')
        assert info is not None
        assert info.step_number == 8
    
    def test_get_step_by_script_unknown(self):
        """Should return None for unknown script."""
        info = get_step_by_script('nonexistent.py')
        assert info is None
    
    def test_all_step_classes_have_matching_info(self):
        """All step classes should have matching info."""
        for step_cls in STEPS:
            info = get_step_by_script(step_cls.script_name)
            assert info is not None, f"No info for {step_cls.script_name}"


class TestStepDependencies:
    """Test step dependency information."""
    
    def test_step_1_has_no_dependencies(self):
        """Step 1 should have no dependencies."""
        assert STEP_INFO[1].requires_steps == []
    
    def test_step_2_requires_step_1(self):
        """Step 2 should require step 1."""
        assert 1 in STEP_INFO[2].requires_steps
    
    def test_step_3_requires_step_2(self):
        """Step 3 should require step 2."""
        assert 2 in STEP_INFO[3].requires_steps
    
    def test_advanced_steps_require_syncro(self):
        """Steps 7-8 should require step 4 (syncro)."""
        assert 4 in STEP_INFO[7].requires_steps
        assert 4 in STEP_INFO[8].requires_steps


class TestDisplayNames:
    """Test display names are user-friendly."""
    
    def test_no_technical_jargon_in_display_names(self):
        """Display names should not contain file extensions."""
        for num, info in STEP_INFO.items():
            assert '.py' not in info.display_name.lower()
            assert '.pkl' not in info.display_name.lower()
    
    def test_display_names_are_in_spanish(self):
        """Display names should be in Spanish."""
        spanish_words = ['de', 'por', 'Global', 'Datos', 'Fuentes', 'Redes']
        all_names = ' '.join(info.display_name for info in STEP_INFO.values())
        # At least some Spanish words should be present
        found = sum(1 for word in spanish_words if word in all_names)
        assert found >= 3, "Display names don't appear to be in Spanish"
    
    def test_button_texts_are_action_verbs(self):
        """Button texts should start with action verbs."""
        action_verbs = ['Ejecutar', 'Consolidar', 'Filtrar', 'Calcular', 'Agregar', 'Identificar']
        for num, info in STEP_INFO.items():
            has_verb = any(info.button_text.startswith(verb) for verb in action_verbs)
            assert has_verb, f"Step {num} button_text doesn't start with action verb"

