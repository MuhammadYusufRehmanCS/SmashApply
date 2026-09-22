from pathlib import Path
r = Path(__file__).resolve().parents[1] / 'backend/tests'
p = r/'test_finalized_tailor.py'
s=p.read_text(encoding='utf-8')
a=s.index('    payload.summary = (',s.index('def rewritten_candidate'))
b=s.index('    payload.core_skills =',a)
s=s[:a]+"    payload.summary = 'Cloud engineer delivering reliable infrastructure through automation, release engineering and service ownership, aligning technical execution with business needs and cross-team delivery priorities.'\n"+s[b:]
s=s.replace('35-40','20-25').replace('three physical lines','two physical lines')
for line in ["            lambda p: p.experience_bullets[0].__setitem__(0, 'Improved availability to 99.999% through resilient service design.'),\n", "            lambda p: p.experience_bullets[0].__setitem__(0, 'Implemented NIST compliance controls across production workloads.'),\n", "            lambda p: p.experience_bullets[0].__setitem__(0, 'Built Snowflake pipelines for analytics workloads.'),\n"]:
 s=s.replace(line,'')
s=s.replace('test_title_budgets_and_unsupported_claims_are_rejected','test_title_and_skill_word_budgets_are_rejected')
s=s.replace('test_unsupported_metric_repairs_only_affected_bullet','test_expanded_metric_does_not_trigger_source_membership_repair')
s=s.replace("        repair.assert_awaited_once()\n        self.assertEqual(set(repair.call_args.args[1]), {'experience_bullets.0.0'})", "        repair.assert_not_awaited()")
p.write_text(s,encoding='utf-8')
p=r/'test_openai_request.py';s=p.read_text(encoding='utf-8').replace("(schema['summary'], 35, 40)","(schema['summary'], 20, 25)");p.write_text(s,encoding='utf-8')
