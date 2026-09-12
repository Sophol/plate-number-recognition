import { HttpErrorResponse } from '@angular/common/http';

/** FastAPI returns {detail: string} or {detail: [{loc, msg, ...}]} for 422. */
export function errorText(err: unknown): string {
  if (!(err instanceof HttpErrorResponse)) return String(err);
  if (err.status === 0) {
    return 'Cannot reach the API at localhost:8000. Is uvicorn running?';
  }
  const detail = err.error?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg}`)
      .join('; ');
  }
  return `${err.status} ${err.statusText}`;
}
