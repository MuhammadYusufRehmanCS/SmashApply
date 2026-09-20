import unittest

from pydantic import ValidationError

from app.services.cv_tailor import (
    SYSTEM_PROMPT, TailoringError, _TailoredPayload, _validate_tailored_payload,
)


class SkillsContractTests(unittest.TestCase):
    def test_pydantic_requires_exactly_three_supplied_items(self):
        for count in (0, 1, 2, 4, 5):
            with self.subTest(count=count), self.assertRaises(ValidationError):
                _TailoredPayload(technical_expertise=["skill"] * count)
        _TailoredPayload(technical_expertise=["Cloud", "Delivery", "Leadership"])
        field = _TailoredPayload.model_json_schema()["properties"]["core_skills"]
        self.assertEqual((field["minItems"], field["maxItems"]), (3, 3))

    def test_validator_checks_three_independently_of_master_category_count(self):
        for source in (None, [], [{"items": "legacy"}] * 4):
            payload = _TailoredPayload(core_skills=["Cloud", "Delivery", "Leadership & Cross-Functional Collaboration: Team ownership"])
            _validate_tailored_payload(payload, False, None, source)
            for count in (0, 2, 4):
                # Model copies/assignments can bypass Pydantic construction.
                payload.technical_expertise = ["skill"] * count
                with self.assertRaisesRegex(TailoringError, "Exactly 3"):
                    _validate_tailored_payload(payload, False, None, source)

    def test_empty_item_is_still_rejected(self):
        payload = _TailoredPayload(technical_expertise=["Cloud", " ", "Leadership"])
        with self.assertRaisesRegex(TailoringError, "empty Technical Expertise"):
            _validate_tailored_payload(payload, False, None, None)

    def test_third_item_cannot_be_another_technical_category(self):
        payload = _TailoredPayload(core_skills=["Cloud: Infrastructure", "Delivery: Releases", "Security: Controls"])
        with self.assertRaisesRegex(TailoringError, "Leadership"):
            _validate_tailored_payload(payload, False, None, None)

    def test_prompt_has_three_item_roles_and_rejection_warning(self):
        for requirement in ("EXACTLY 3 items", "Bullet 1 = Dynamic Domain Skill 1",
                            "Bullet 2 = Dynamic Domain Skill 2",
                            "Bullet 3 = Leadership & Cross-Functional Collaboration",
                            "will trigger payload rejection"):
            self.assertIn(requirement, SYSTEM_PROMPT)
        self.assertNotIn('"Category 4 items"', SYSTEM_PROMPT)
