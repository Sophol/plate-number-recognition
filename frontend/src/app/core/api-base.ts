/** Same-origin API prefix.
 *
 *  nginx serves the built app at / and proxies /api/ to the FastAPI process, so
 *  a relative base means the browser never makes a cross-origin request and CORS
 *  never applies. It also means the same build works on any hostname - no
 *  rebuild to move between staging and production.
 *
 *  For `ng serve`, proxy.conf.json forwards /api to the local API so development
 *  behaves the same way.
 */
export const API_BASE = '/api';
