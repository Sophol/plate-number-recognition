import { Injectable, computed, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';
import { Me, Role, ROLE_RANK, Token } from './models';
import { API_BASE } from './api-base';

const TOKEN_KEY = 'anpr_token';
const USER_KEY = 'anpr_user';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly _token = signal<string | null>(this.readToken());
  private readonly _user = signal<Me | null>(this.readUser());

  readonly user = this._user.asReadonly();
  readonly isLoggedIn = computed(() => this._token() !== null);
  readonly role = computed(() => this._user()?.role ?? null);

  /** Writes require list_editor or higher; reads only need viewer. */
  readonly canEdit = computed(() => {
    const r = this.role();
    return r !== null && ROLE_RANK[r] >= ROLE_RANK.list_editor;
  });

  constructor(private http: HttpClient) {}

  get token(): string | null {
    return this._token();
  }

  /** The API uses OAuth2 password flow, so this must be form-encoded, not JSON. */
  login(username: string, password: string): Observable<Token> {
    const body = new URLSearchParams();
    body.set('username', username);
    body.set('password', password);

    return this.http
      .post<Token>(`${API_BASE}/auth/token`, body.toString(), {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      })
      .pipe(tap((t) => this.setToken(t.access_token)));
  }

  loadMe(): Observable<Me> {
    return this.http
      .get<Me>(`${API_BASE}/auth/me`)
      .pipe(tap((me) => this.setUser(me)));
  }

  logout(): void {
    this._token.set(null);
    this._user.set(null);
    try {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    } catch {
      /* storage unavailable (private window) - in-memory state is enough */
    }
  }

  private setToken(token: string): void {
    this._token.set(token);
    try {
      localStorage.setItem(TOKEN_KEY, token);
    } catch {}
  }

  private setUser(me: Me): void {
    this._user.set(me);
    try {
      localStorage.setItem(USER_KEY, JSON.stringify(me));
    } catch {}
  }

  private readToken(): string | null {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  }

  private readUser(): Me | null {
    try {
      const raw = localStorage.getItem(USER_KEY);
      return raw ? (JSON.parse(raw) as Me) : null;
    } catch {
      return null;
    }
  }
}
