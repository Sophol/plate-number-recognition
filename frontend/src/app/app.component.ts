import { Component, inject } from '@angular/core';
import { Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { AuthService } from './core/auth.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    @if (auth.isLoggedIn()) {
      <nav class="topbar">
        <span class="brand">ANPR</span>
        <a routerLink="/live" routerLinkActive="active">Live</a>
        <a routerLink="/plate-reads" routerLinkActive="active">Plate reads</a>
        <a routerLink="/cameras" routerLinkActive="active">Cameras</a>
        <a routerLink="/vehicles" routerLinkActive="active">Vehicles</a>
        <span class="spacer"></span>
        <span class="who">
          {{ auth.user()?.username }}
          <span class="tag role">{{ auth.user()?.role }}</span>
        </span>
        <button class="ghost" (click)="logout()">Sign out</button>
      </nav>
    }
    <main class="content">
      <router-outlet />
    </main>
  `,
})
export class AppComponent {
  auth = inject(AuthService);
  private router = inject(Router);

  logout(): void {
    this.auth.logout();
    this.router.navigate(['/login']);
  }
}
