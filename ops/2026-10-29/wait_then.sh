#!/usr/bin/env bash
# Wait until every condition holds, then run the command. A condition is a log file that must contain a line
# matching PATTERN (FILE:PATTERN, default QUEUE DONE) or pid:N, a process that must have exited.
# Usage: wait_then.sh COND... -- COMMAND...
while [ "$1" != "--" ]; do
  case $1 in
    pid:*) while kill -0 ${1#pid:} 2>/dev/null; do sleep 60; done ;;
    *:*) until grep -qs "${1#*:}" ${1%%:*}; do sleep 60; done ;;
    *) until grep -qs "QUEUE DONE" $1; do sleep 60; done ;;
  esac
  shift
done
shift; exec "$@"
