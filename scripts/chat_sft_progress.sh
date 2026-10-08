#!/usr/bin/env bash
set -u

pid="${1:?pid required}"
log="${2:?log required}"
total="${3:?total steps required}"
checkpoint="${4:---}"
tokens_per_step="${5:-256}"
cyan=$'\033[38;5;81m'; white=$'\033[1;97m'; muted=$'\033[38;5;245m'
green=$'\033[38;5;120m'; reset=$'\033[0m'

while kill -0 "$pid" 2>/dev/null; do
  line=$(grep '"step"' "$log" | tail -1)
  step=$(printf '%s' "$line" | sed -n 's/.*"step": \([0-9]*\).*/\1/p')
  loss=$(printf '%s' "$line" | sed -n 's/.*"loss": \([0-9.eE+-]*\).*/\1/p')
  elapsed_s=$(printf '%s' "$line" | sed -n 's/.*"elapsed_seconds": \([0-9.eE+-]*\).*/\1/p')
  [[ "$step" =~ ^[0-9]+$ ]] || step=0
  [[ -n "$loss" ]] || loss="--"
  pct=$((step * 100 / total)); filled=$((pct / 2)); empty=$((50 - filled))
  bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
  rest=$(printf '%*s' "$empty" '' | tr ' ' '-')
  speed='--'; eta='--:--:--'
  if [[ -n "$elapsed_s" && "$elapsed_s" != 0 ]]; then
    speed=$(awk -v t="$step" -v e="$elapsed_s" -v b="$tokens_per_step" 'BEGIN {printf "%.1f", t*b/e}')
    remaining=$(awk -v r="$((total-step))" -v e="$elapsed_s" -v t="$step" 'BEGIN {if (t>0) printf "%d", r*e/t; else print 0}')
    eta=$(printf '%02d:%02d:%02d' $((remaining/3600)) $(((remaining/60)%60)) $((remaining%60)))
  fi
  tokens=$(awk -v s="$step" -v b="$tokens_per_step" 'BEGIN {printf "%.1fK", s*b/1000}')
  clear
  printf '%s  CODEXA NUMPY%s   %sCHAT SFT%s\n' "$cyan" "$reset" "$white" "$reset"
  printf '%s  ------------------------------------------------------------------------%s\n\n' "$muted" "$reset"
  printf '  %s%s%s  %s%3d%%%s    ETA %s%s%s    %sRUNNING%s\n\n' "$green" "$bar$rest" "$reset" "$white" "$pct" "$reset" "$white" "$eta" "$reset" "$green" "$reset"
  printf '  %sSTEPS%s       %s%s / %s%s\n' "$muted" "$reset" "$white" "$step" "$total" "$reset"
  printf '  %sSPEED%s       %s%s tok/s%s\n' "$muted" "$reset" "$white" "$speed" "$reset"
  printf '  %sTRAIN LOSS%s  %s%s%s\n' "$muted" "$reset" "$white" "$loss" "$reset"
  printf '  %sVAL LOSS%s    %spending completion%s\n' "$muted" "$reset" "$white" "$reset"
  printf '  %sTOKENS%s      %s%s%s\n' "$muted" "$reset" "$white" "$tokens" "$reset"
  printf '  %sELAPSED%s     %s%s seconds%s\n\n' "$muted" "$reset" "$white" "${elapsed_s:---}" "$reset"
  printf '  %sCHECKPOINT%s\n  %s%s%s\n\n' "$cyan" "$reset" "$muted" "$checkpoint" "$reset"
  printf '  %sCUDA FP32  |  refresh 2s  |  PID %s%s\n' "$muted" "$pid" "$reset"
  sleep 2
done
clear
printf '%s  CHAT SFT COMPLETE%s\n' "$green" "$reset"
exec zsh
