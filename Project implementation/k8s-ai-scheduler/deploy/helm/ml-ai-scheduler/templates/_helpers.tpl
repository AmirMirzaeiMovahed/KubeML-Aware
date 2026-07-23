{{- define "ml-ai-scheduler.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "ml-ai-scheduler.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name (include "ml-ai-scheduler.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "ml-ai-scheduler.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
app.kubernetes.io/name: {{ include "ml-ai-scheduler.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: scheduler
{{- end }}

{{- define "ml-ai-scheduler.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ml-ai-scheduler.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: scheduler
{{- end }}

{{- define "ml-ai-scheduler.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "ml-ai-scheduler.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- required "serviceAccount.name is required when serviceAccount.create=false" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "ml-ai-scheduler.image" -}}
{{- if .Values.image.digest -}}
{{ printf "%s@%s" .Values.image.repository .Values.image.digest }}
{{- else -}}
{{ printf "%s:%s" .Values.image.repository .Values.image.tag }}
{{- end -}}
{{- end }}

{{- define "ml-ai-scheduler.validate" -}}
{{- if not (has .Values.mode (list "production" "reproduction")) -}}
{{- fail "mode must be production or reproduction" -}}
{{- end -}}
{{- if not (has .Values.scheduler.pacingMode (list "none" "fixed" "adaptive")) -}}
{{- fail "scheduler.pacingMode must be none, fixed, or adaptive" -}}
{{- end -}}
{{- if lt (int .Values.replicaCount) 1 -}}
{{- fail "replicaCount must be at least 1" -}}
{{- end -}}
{{- if gt (int .Values.replicaCount) 1 -}}
{{- fail "multiple replicas require leader election; keep replicaCount=1" -}}
{{- end -}}
{{- if and (eq .Values.mode "reproduction") (not .Values.scheduler.targetNode) -}}
{{- fail "reproduction mode requires scheduler.targetNode" -}}
{{- end -}}
{{- if ne (not (empty .Values.experiment.runId)) (gt (int .Values.scheduler.expectedCount) 0) -}}
{{- fail "experiment.runId and scheduler.expectedCount must be set together" -}}
{{- end -}}
{{- if and (not .Values.image.digest) (eq .Values.image.tag "latest") -}}
{{- fail "image tag latest is forbidden; use an immutable version or digest" -}}
{{- end -}}
{{- if and .Values.image.digest (not (regexMatch "^sha256:[a-f0-9]{64}$" .Values.image.digest)) -}}
{{- fail "image.digest must be a full lowercase sha256 digest" -}}
{{- end -}}
{{- end }}
