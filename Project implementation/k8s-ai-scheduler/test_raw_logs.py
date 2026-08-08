"""Test to show the exact raw string the controller receives from K8s API"""

from scheduler.execution import parse_execution_marker

# This is what the K8s Python client ACTUALLY returns for a completed pod
# Type: <class 'str'>
# It's a STRING that contains the repr() of the bytes object!
# Notice: starts with "b'" and ends with "'" and has \\n for newlines
logs_raw_from_api = 'b\'{"config":{"C":10,"G":64.0,"M":256,"P":1,"R":0.5,"T":60.0,"job_id":"workload-1","seed":228711551},"event":"JOB_CONFIG","job_id":"workload-1","timestamp":1786196225.8616173}\\n{"blas_runtime":{"expected_threads":1,"libraries":[{"architecture":"Haswell","internal_api":"openblas","num_threads":1,"prefix":"libscipy_openblas","threading_layer":"pthreads","user_api":"blas","version":"0.3.30"}]},"blas_threads":1,"estimated_T":60.0,"estimated_T_error_seconds":59.7632,"event":"INITIALIZATION_COMPLETED","job_id":"workload-1","matrix_shape":[256,256],"partition_rows":[[0,256]],"seed":228711551,"timestamp":1786196226.1888156,"work_model":{"checkpoint_bytes":262144,"checkpoint_count":0,"checkpoint_seconds":0.0,"convergence_steps":8,"estimated_training_seconds":0.2368,"gradient_bytes":67108864,"gradient_update_seconds":0.2048,"matrix_compute_seconds":0.032,"model_version":"2.0","partition_sync_seconds":0.0,"planned_steps":8,"step_budget":237,"termination_reason":"converged"}}\\n{"event":"EXECUTION_STARTED","job_id":"workload-1","timestamp":1786196226.188972}\\n{"event":"CONVERGED","job_id":"workload-1","loss":0.01831563888873418,"step":8,"timestamp":1786196226.352641}\\n{"blas_library_count":1,"blas_threads":1,"checkpoint_bytes":262144,"checkpoint_count":0,"checkpoint_seconds":0.0,"convergence_steps":8,"duration_seconds":0.16384577751159668,"event":"EXECUTION_COMPLETED","final_loss":0.01831563888873418,"gradient_bytes":67108864,"job_id":"workload-1","step_budget":237,"steps":8,"termination_reason":"converged","timestamp":1786196226.3528178,"work_model_version":"2.0"}\\n\''

print("=== RAW STRING FROM K8s API (what controller actually receives) ===")
print(f"Type: {type(logs_raw_from_api)}")
print(f"Length: {len(logs_raw_from_api)}")
print(f"First 100 chars: {repr(logs_raw_from_api[:100])}")
print(f"Last 100 chars: {repr(logs_raw_from_api[-100:])}")
print()

result = parse_execution_marker(logs_raw_from_api, expected_job_id="workload-1")
print(f"parse_execution_marker result: {result}")
print()

# What it SHOULD be (decoded bytes)
logs_actual = logs_raw_from_api[2:-1].encode().decode("unicode_escape")
print("=== WHAT IT SHOULD BE (after proper decoding) ===")
print(f"Type: {type(logs_actual)}")
result2 = parse_execution_marker(logs_actual, expected_job_id="workload-1")
print(f"parse_execution_marker result: {result2}")
