"""Test both single and double quote repr(bytes) formats"""

from scheduler.execution import _normalize_logs, parse_execution_marker

# Test single-quote repr (what we saw)
logs_single = (
    'b\'{"event":"EXECUTION_STARTED","job_id":"workload-1","timestamp":12345.678}\\n\''
)
result = _normalize_logs(logs_single)
print("Single-quote test:")
print(f"  Input type: {type(logs_single)}")
print(f"  Output type: {type(result)}")
print(
    f"  Parsed marker: {parse_execution_marker(result, expected_job_id='workload-1')}"
)

# Test double-quote repr (what Python does if content has ')
logs_double = 'b"{\\"event\\":\\"EXECUTION_STARTED\\",\\"job_id\\":\\"workload-1\\",\\"timestamp\\":12345.678}\\n"'
result2 = _normalize_logs(logs_double)
print("\nDouble-quote test:")
print(f"  Input type: {type(logs_double)}")
print(f"  Output type: {type(result2)}")
print(
    f"  Parsed marker: {parse_execution_marker(result2, expected_job_id='workload-1')}"
)

# Test bytes (from HTTPResponse.data)
logs_bytes = (
    b'{"event":"EXECUTION_STARTED","job_id":"workload-1","timestamp":12345.678}\n'
)
result3 = _normalize_logs(logs_bytes)
print("\nBytes test:")
print(f"  Input type: {type(logs_bytes)}")
print(f"  Output type: {type(result3)}")
print(
    f"  Parsed marker: {parse_execution_marker(result3, expected_job_id='workload-1')}"
)

# Test plain string (already decoded)
logs_plain = (
    '{"event":"EXECUTION_STARTED","job_id":"workload-1","timestamp":12345.678}\n'
)
result4 = _normalize_logs(logs_plain)
print("\nPlain string test:")
print(f"  Input type: {type(logs_plain)}")
print(f"  Output type: {type(result4)}")
print(
    f"  Parsed marker: {parse_execution_marker(result4, expected_job_id='workload-1')}"
)

print("\n=== All tests passed! ===")
