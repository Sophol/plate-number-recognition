import { Component, inject, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { AuthService } from '../../core/auth.service';
import { Camera, CameraCreate } from '../../core/models';
import { errorText } from '../../core/error-text';

/** A hostname the camera worker will never resolve, e.g. rtsp://x or rtsp://demo.
 *  These are the seeded placeholders that make the worker log rtsp_connect_failed. */
function isPlaceholder(url: string): boolean {
  const host = url.replace(/^rtsp:\/\//i, '').split(/[/:?]/)[0];
  return host.length > 0 && !host.includes('.') && host !== 'localhost';
}

@Component({
  selector: 'app-cameras',
  standalone: true,
  imports: [FormsModule],
  template: `
    <header class="page-head">
      <h2>Cameras</h2>
      <label class="inline">
        <input type="checkbox" [(ngModel)]="activeOnly" (change)="load()" />
        active only
      </label>
      <button class="ghost" (click)="load()">Refresh</button>
    </header>

    @if (error()) { <div class="alert error">{{ error() }}</div> }

    @if (auth.canEdit()) {
      <form class="card row-form" (ngSubmit)="create()">
        <div>
          <label>Name *</label>
          <input name="name" [(ngModel)]="draft.name" required />
        </div>
        <div class="grow">
          <label>RTSP URL *</label>
          <input name="rtsp" [(ngModel)]="draft.rtsp_url" placeholder="rtsp://host:554/stream" required />
        </div>
        <div>
          <label>Site</label>
          <input name="site" [(ngModel)]="draft.site_id" />
        </div>
        <div>
          <label>Direction</label>
          <input name="dir" [(ngModel)]="draft.direction" />
        </div>
        <button type="submit" [disabled]="busy() || !canSubmit">Add camera</button>
      </form>
    } @else {
      <div class="alert info">Your role is read-only. Creating cameras needs list_editor or higher.</div>
    }

    <table class="data">
      <thead>
        <tr><th>Name</th><th>RTSP URL</th><th>Site</th><th>Direction</th><th>Active</th><th></th></tr>
      </thead>
      <tbody>
        @for (c of cameras(); track c.id) {
          <tr>
            <td>
              @if (c.name.trim()) {
                {{ c.name }}
              } @else {
                <span class="muted">camera {{ c.id.slice(0, 8) }}</span>
                <span class="tag warn" title="This camera has no name.">unnamed</span>
              }
            </td>
            <td class="mono">
              {{ c.rtsp_url }}
              @if (isPlaceholder(c.rtsp_url)) {
                <span class="tag warn" title="This hostname will not resolve; the camera worker will retry forever.">unreachable</span>
              }
            </td>
            <td>{{ c.site_id || '—' }}</td>
            <td>{{ c.direction || '—' }}</td>
            <td>
              <span class="tag" [class.ok]="c.is_active" [class.off]="!c.is_active">
                {{ c.is_active ? 'active' : 'inactive' }}
              </span>
            </td>
            <td class="actions">
              @if (auth.canEdit()) {
                <button class="ghost" (click)="toggle(c)">
                  {{ c.is_active ? 'Deactivate' : 'Activate' }}
                </button>
                <button class="ghost" (click)="rename(c)">Rename</button>
                <button class="ghost" (click)="editUrl(c)">Edit URL</button>
              }
            </td>
          </tr>
        } @empty {
          <tr><td colspan="6" class="muted">No cameras.</td></tr>
        }
      </tbody>
    </table>
  `,
})
export class CamerasComponent implements OnInit {
  private api = inject(ApiService);
  auth = inject(AuthService);

  cameras = signal<Camera[]>([]);
  error = signal('');
  busy = signal(false);
  activeOnly = false;
  draft: CameraCreate = { name: '', rtsp_url: '', site_id: '', direction: '' };

  isPlaceholder = isPlaceholder;

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.error.set('');
    this.api.listCameras(this.activeOnly ? true : undefined).subscribe({
      next: (c) => this.cameras.set(c),
      error: (e) => this.error.set(errorText(e)),
    });
  }

  /** The API rejects blank/whitespace names, so block the submit locally too. */
  get canSubmit(): boolean {
    return this.draft.name.trim().length > 0 && this.draft.rtsp_url.trim().length > 0;
  }

  create(): void {
    if (!this.canSubmit) return;
    this.busy.set(true);
    this.error.set('');
    const body: CameraCreate = {
      name: this.draft.name.trim(),
      rtsp_url: this.draft.rtsp_url.trim(),
      site_id: this.draft.site_id?.trim() || null,
      direction: this.draft.direction?.trim() || null,
    };
    this.api.createCamera(body).subscribe({
      next: () => {
        this.busy.set(false);
        this.draft = { name: '', rtsp_url: '', site_id: '', direction: '' };
        this.load();
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(errorText(e));
      },
    });
  }

  toggle(c: Camera): void {
    this.api.updateCamera(c.id, { is_active: !c.is_active }).subscribe({
      next: () => this.load(),
      error: (e) => this.error.set(errorText(e)),
    });
  }

  rename(c: Camera): void {
    const next = prompt(`Name for this camera`, c.name)?.trim();
    if (!next || next === c.name) return;
    this.api.updateCamera(c.id, { name: next }).subscribe({
      next: () => this.load(),
      error: (e) => this.error.set(errorText(e)),
    });
  }

  editUrl(c: Camera): void {
    const next = prompt(`RTSP URL for ${c.name}`, c.rtsp_url)?.trim();
    if (!next || next === c.rtsp_url) return;
    this.api.updateCamera(c.id, { rtsp_url: next }).subscribe({
      next: () => this.load(),
      error: (e) => this.error.set(errorText(e)),
    });
  }
}
