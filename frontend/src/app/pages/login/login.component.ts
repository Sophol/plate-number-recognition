import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/auth.service';
import { errorText } from '../../core/error-text';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="login-wrap">
      <form class="card login-card" (ngSubmit)="submit()">
        <h1>ANPR Test Console</h1>
        <p class="muted">Sign in to manage cameras, vehicles and plate reads</p>

        <label>Username</label>
        <input name="username" [(ngModel)]="username" autocomplete="username" required />

        <label>Password</label>
        <input
          name="password"
          type="password"
          [(ngModel)]="password"
          autocomplete="current-password"
          required
        />

        @if (error()) {
          <div class="alert error">{{ error() }}</div>
        }

        <button type="submit" [disabled]="busy()">
          {{ busy() ? 'Signing in…' : 'Sign in' }}
        </button>
      </form>
    </div>
  `,
  styles: [`
    .login-wrap { display: grid; place-items: center; min-height: 80vh; }
    .login-card { width: min(380px, 100%); }
    h1 { margin: 0 0 4px; font-size: 20px; }
    label { display: block; margin-top: 12px; }
    button { margin-top: 16px; width: 100%; }
  `],
})
export class LoginComponent {
  private auth = inject(AuthService);
  private router = inject(Router);

  username = 'admin';
  password = '';
  busy = signal(false);
  error = signal('');

  submit(): void {
    this.busy.set(true);
    this.error.set('');
    this.auth.login(this.username, this.password).subscribe({
      next: () =>
        this.auth.loadMe().subscribe({
          next: () => {
            this.busy.set(false);
            this.router.navigate(['/plate-reads']);
          },
          error: (e) => {
            this.busy.set(false);
            this.error.set(errorText(e));
          },
        }),
      error: (e) => {
        this.busy.set(false);
        this.error.set(errorText(e));
      },
    });
  }
}
