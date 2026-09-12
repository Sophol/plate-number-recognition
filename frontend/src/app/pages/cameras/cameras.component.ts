import { Component, inject, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { AuthService } from '../../core/auth.service';
import { Camera, CameraCreate, CameraUpdate } from '../../core/models';
import { errorText } from '../../core/error-text';

/** A hostname the camera worker will never resolve, e.g. rtsp://x or rtsp://demo.
 *  These are the seeded placeholders that make the worker log rtsp_connect_failed. */
function isPlaceholder(url: string): boolean {
  const host = url.replace(/^rtsp:\/\//i, '').split(/[/:?]/)[0];
  return host.length > 0 && !host.includes('.') && host !== 'localhost';
}

/** rtsp://user:secret@host -> rtsp://user:•••@host for display. The full URL
 *  is still there in the edit form; it just should not sit on screen all day. */
function maskUrl(url: string): string {
  return url.replace(/^(rtsp:\/\/[^:@/]+:)[^@/]+@/i, '$1•••@');
}

const DIRECTIONS = ['in', 'out'] as const;

/** Everything on a camera that can be edited, as the form holds it. */
interface CameraForm {
  name: string;
  rtsp_url: string;
  site_id: string;
  direction: string;
  is_active: boolean;
}

function toForm(c: Camera): CameraForm {
  return {
    name: c.name,
    rtsp_url: c.rtsp_url,
    site_id: c.site_id ?? '',
    direction: c.direction ?? '',
    is_active: c.is_active,
  };
}

const EMPTY: CameraForm = { name: '', rtsp_url: '', site_id: '', direction: '', is_active: true };

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
          <input name="rtsp" [(ngModel)]="draft.rtsp_url" placeholder="rtsp://user:pass@host:554/stream" required />
        </div>
        <div>
          <label>Site</label>
          <input name="site" [(ngModel)]="draft.site_id" placeholder="gate-06" />
        </div>
        <div>
          <label>Direction</label>
          <select name="dir" [(ngModel)]="draft.direction">
            <option value="">—</option>
            @for (d of directions; track d) { <option [value]="d">{{ d }}</option> }
          </select>
        </div>
        <label class="inline">
          <input type="checkbox" name="active" [(ngModel)]="draft.is_active" /> active
        </label>
        <button type="submit" [disabled]="busy() || !canSubmit(draft)">Add camera</button>
      </form>
    } @else {
      <div class="alert info">Your role is read-only. Editing cameras needs list_editor or higher.</div>
    }

    <table class="data">
      <thead>
        <tr><th>Name</th><th>RTSP URL</th><th>Site</th><th>Direction</th><th>Active</th><th></th></tr>
      </thead>
      <tbody>
        @for (c of cameras(); track c.id) {
          @if (editing()?.id === c.id) {
            <tr class="editing">
              <td><input name="e-name" [(ngModel)]="form.name" required /></td>
              <td><input name="e-rtsp" class="mono wide" [(ngModel)]="form.rtsp_url" required /></td>
              <td><input name="e-site" class="narrow" [(ngModel)]="form.site_id" /></td>
              <td>
                <select name="e-dir" [(ngModel)]="form.direction">
                  <option value="">—</option>
                  @for (d of directions; track d) { <option [value]="d">{{ d }}</option> }
                </select>
              </td>
              <td><label class="inline"><input type="checkbox" name="e-active" [(ngModel)]="form.is_active" /> active</label></td>
              <td class="actions">
                <button (click)="save(c)" [disabled]="busy() || !canSubmit(form) || !dirty(c)">Save</button>
                <button class="ghost" (click)="cancel()" [disabled]="busy()">Cancel</button>
              </td>
            </tr>
          } @else {
            <tr>
              <td>
                @if (c.name.trim()) {
                  {{ c.name }}
                } @else {
                  <span class="muted">camera {{ c.id.slice(0, 8) }}</span>
                  <span class="tag warn" title="This camera has no name.">unnamed</span>
                }
              </td>
              <td class="mono" [title]="c.rtsp_url">
                {{ maskUrl(c.rtsp_url) }}
                @if (isPlaceholder(c.rtsp_url)) {
                  <span class="tag warn" title="This hostname will not resolve; the camera worker will retry forever.">unreachable</span>
                }
              </td>
              <td>{{ c.site_id || '—' }}</td>
              <td>
                @if (c.direction) { <span class="tag" [class.ok]="c.direction === 'in'" [class.warn]="c.direction === 'out'">{{ c.direction }}</span> }
                @else { — }
              </td>
              <td>
                <span class="tag" [class.ok]="c.is_active" [class.off]="!c.is_active">
                  {{ c.is_active ? 'active' : 'inactive' }}
                </span>
              </td>
              <td class="actions">
                @if (auth.canEdit()) {
                  <button class="ghost" (click)="edit(c)" [disabled]="busy()">Edit</button>
                  <button class="ghost" (click)="toggle(c)" [disabled]="busy()">
                    {{ c.is_active ? 'Deactivate' : 'Activate' }}
                  </button>
                }
              </td>
            </tr>
          }
        } @empty {
          <tr><td colspan="6" class="muted">No cameras.</td></tr>
        }
      </tbody>
    </table>
    @if (auth.canEdit()) {
      <p class="muted small">
        Changes to a camera's URL or active flag take effect when the pipeline service restarts.
      </p>
    }
  `,
  styles: [`
    tr.editing td { vertical-align: middle; }
    tr.editing input, tr.editing select { width: 100%; box-sizing: border-box; }
    tr.editing input.narrow { width: 7rem; }
    tr.editing input.wide { min-width: 22rem; }
  `],
})
export class CamerasComponent implements OnInit {
  private api = inject(ApiService);
  auth = inject(AuthService);

  cameras = signal<Camera[]>([]);
  error = signal('');
  busy = signal(false);
  editing = signal<Camera | null>(null);
  activeOnly = false;
  draft: CameraForm = { ...EMPTY };
  form: CameraForm = { ...EMPTY };

  readonly directions = DIRECTIONS;
  isPlaceholder = isPlaceholder;
  maskUrl = maskUrl;

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
  canSubmit(f: CameraForm): boolean {
    return f.name.trim().length > 0 && f.rtsp_url.trim().length > 0;
  }

  create(): void {
    if (!this.canSubmit(this.draft)) return;
    const body: CameraCreate = {
      name: this.draft.name.trim(),
      rtsp_url: this.draft.rtsp_url.trim(),
      site_id: this.draft.site_id.trim() || null,
      direction: this.draft.direction || null,
      is_active: this.draft.is_active,
    };
    this.run(this.api.createCamera(body), () => (this.draft = { ...EMPTY }));
  }

  edit(c: Camera): void {
    this.editing.set(c);
    this.form = toForm(c);
  }

  cancel(): void {
    this.editing.set(null);
  }

  /** Only the fields that actually changed go in the PATCH. */
  changes(c: Camera): CameraUpdate {
    const out: CameraUpdate = {};
    const name = this.form.name.trim();
    const url = this.form.rtsp_url.trim();
    const site = this.form.site_id.trim() || null;
    const dir = this.form.direction || null;
    if (name !== c.name) out.name = name;
    if (url !== c.rtsp_url) out.rtsp_url = url;
    if (site !== (c.site_id ?? null)) out.site_id = site;
    if (dir !== (c.direction ?? null)) out.direction = dir;
    if (this.form.is_active !== c.is_active) out.is_active = this.form.is_active;
    return out;
  }

  dirty(c: Camera): boolean {
    return Object.keys(this.changes(c)).length > 0;
  }

  save(c: Camera): void {
    const patch = this.changes(c);
    if (!Object.keys(patch).length) return this.cancel();
    this.run(this.api.updateCamera(c.id, patch), () => this.editing.set(null));
  }

  toggle(c: Camera): void {
    this.run(this.api.updateCamera(c.id, { is_active: !c.is_active }));
  }

  private run(call: { subscribe: Function }, after?: () => void): void {
    this.busy.set(true);
    this.error.set('');
    call.subscribe({
      next: () => {
        this.busy.set(false);
        after?.();
        this.load();
      },
      error: (e: unknown) => {
        this.busy.set(false);
        this.error.set(errorText(e));
      },
    });
  }
}
