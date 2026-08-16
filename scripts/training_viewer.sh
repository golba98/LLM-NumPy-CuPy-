#!/usr/bin/env bash
set -u

run_dir="${1:?run directory required}"
total_steps="${2:?total steps required}"
pid="${3:-}"
checkpoint="${4:---}"
metrics="$run_dir/target.csv"
started=$(date +%s)

cyan=$'\033[38;5;81m'; white=$'\033[1;97m'; muted=$'\033[38;5;245m'
green=$'\033[38;5;120m'; yellow=$'\033[38;5;221m'; reset=$'\033[0m'

while :; do
  step=0; train='--'; val='--'; speed='--'; tokens='--'; step_time='--'
  if [[ -f "$metrics" ]]; then
    last=$(tail -n 1 "$metrics")
    IFS=',' read -r step _ tokens train _ _ val _ _ _ _ speed step_time <<< "$last"
    [[ "$step" =~ ^[0-9]+$ ]] || step=0
    [[ -n "$train" ]] || train='--'
    [[ -n "$val" ]] || val='--'
    [[ -n "$speed" ]] || speed='--'
    [[ -n "$tokens" ]] || tokens='--'
    [[ -n "$step_time" ]] || step_time='--'
  fi
  [[ "$train" == '--' ]] || train=$(awk -v v="$train" 'BEGIN{printf "%.3f",v}')
  [[ "$val" == '--' ]] || val=$(awk -v v="$val" 'BEGIN{printf "%.3f",v}')
  [[ "$speed" == '--' ]] || speed=$(awk -v v="$speed" 'BEGIN{printf "%.2f",v}')
  [[ "$tokens" == '--' ]] || tokens=$(awk -v v="$tokens" 'BEGIN{printf "%.1fK",v/1000}')
  alive=0; [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1
  if (( step >= total_steps )); then status='COMPLETE'; pct=100; status_color=$green
  elif (( alive )); then status='RUNNING'; pct=$((step*100/total_steps)); status_color=$cyan
  else status='STOPPED'; pct=$((step*100/total_steps)); status_color=$yellow; fi
  (( pct > 100 )) && pct=100
  eta='--:--:--'
  if [[ "$step_time" != '--' && "$step" -lt "$total_steps" ]]; then
    seconds=$(awk -v r="$((total_steps-step))" -v s="$step_time" 'BEGIN{printf "%d",r*s}')
    eta=$(printf '%02d:%02d:%02d' $((seconds/3600)) $(((seconds/60)%60)) $((seconds%60)))
  fi
  filled=$((pct*50/100)); empty=$((50-filled))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#'); rest=$(printf '%*s' "$empty" '' | tr ' ' '-')
  now=$(date +%s); elapsed=$((now-started))

  printf '\033[2J\033[H'
  printf '%s  CODEXA NUMPY%s   %sTARGET PRETRAINING%s\n' "$cyan" "$reset" "$white" "$reset"
  printf '%s  ------------------------------------------------------------------------%s\n\n' "$muted" "$reset"
  printf '  %s%s%s  %s%3d%%%s    ETA %s%s%s    %s%s%s\n\n' "$status_color" "$bar$rest" "$reset" "$white" "$pct" "$reset" "$white" "$eta" "$reset" "$status_color" "$status" "$reset"
  printf '  %sSTEPS%s       %s%s%s\n' "$muted" "$reset" "$white" "$step / $total_steps" "$reset"
  printf '  %sSPEED%s       %s%s tok/s%s\n' "$muted" "$reset" "$white" "$speed" "$reset"
  printf '  %sTRAIN LOSS%s  %s%s%s\n' "$muted" "$reset" "$white" "$train" "$reset"
  printf '  %sVAL LOSS%s    %s%s%s\n' "$muted" "$reset" "$white" "$val" "$reset"
  printf '  %sTOKENS%s      %s%s%s\n' "$muted" "$reset" "$white" "$tokens" "$reset"
  printf '  %sELAPSED%s     %s%02d:%02d:%02d%s\n\n' "$muted" "$reset" "$white" $((elapsed/3600)) $(((elapsed/60)%60)) $((elapsed%60)) "$reset"
  printf '  %sCHECKPOINT%s\n  %s%s%s\n\n' "$cyan" "$reset" "$muted" "$checkpoint" "$reset"
  printf '  %sLOG%s\n  %s%s%s\n\n' "$cyan" "$reset" "$muted" "$metrics" "$reset"
  printf '  %sCUDA  FP16  |  refresh 2s%s\n' "$muted" "$reset"
  if [[ "$status" != RUNNING ]]; then printf '\n  %sRun is %s. Press Enter to close.%s' "$status_color" "$status" "$reset"; read -r _; break; fi
  sleep 2
done
