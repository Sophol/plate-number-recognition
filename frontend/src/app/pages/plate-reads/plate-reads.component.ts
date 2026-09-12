import { Component, inject, OnInit, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { AuthService } from '../../core/auth.service';
import { Camera, PlateRead, PlateReadCreate, PlateReadQuery } from '../../core/models';
import { errorText } from '../../core/error-text';

@Component({
  selector: 'app-plate-reads',
  standalone: true,
  imports: [FormsModule, DatePipe, DecimalPipe],
  template: `
    <header class="page-head">
      <h2>Plate reads</h2>
      <button class="ghost" (click)="load()">Refresh</button>
    </header>

    @if (error()) { <div class="alert error">{{ error() }}</div> }

    <form class="card row-form" (ngSubmit)="load()">
      <div>
        <label>Camera</label>
        <select name="cam" [(ngModel)]="q.camera_id">
          <option value="">any</option>
          @for (c of cameras(); track c.id) {
            <option [value]="c.id">{{ c.name }}</option>
          }
        </select>
      </div>
      <div>
        <label>Plate text</label>
        <input name="plate" [(ngModel)]="q.plate_text" placeholder="partial match" />
      </div>
      <div>
        <label>Since</label>
        <input name="since" type="datetime-local" [(ngModel)]="sinceLocal" />
      </div>
      <div>
        <label>Until</label>
        <input name="until" type="datetime-local" [(ngModel)]="untilLocal" />
      </div>
      <div>
        <label>Limit</label>
        <input name="limit" type="number" min="1" [(ngModel)]="q.limit" class="narrow" />
      </div>
      <label class="inline">
        <input type="checkbox" name="active" [(ngModel)]="q.only_active_model" />
        active model only
      </label>
      <button type="submit">Apply filters</button>
      <button type="button" class="ghost" (click)="reset()">Reset</button>
    </form>

    @if (auth.canEdit()) {
      <details class="card">
        <summary>Insert a synthetic read (useful without a live camera)</summary>
        <form class="row-form" (ngSubmit)="create()">
          <div>
            <label>Camera *</label>
            <select name="ccam" [(ngModel)]="draft.camera_id" required>
              <option value="">choose…</option>
              @for (c of cameras(); track c.id) {
                <option [value]="c.id">{{ c.name }}</option>
              }
            </select>
          </div>
          <div>
            <label>Plate text *</label>
            <input name="cplate" [(ngModel)]="draft.plate_text" required />
          </div>
          <div>
            <label>Confidence *</label>
            <input name="cconf" type="number" step="0.01" min="0" max="1"
                   [(ngModel)]="draft.confidence" class="narrow" required />
          </div>
          <div>
            <label>Province</label>
            <input name="cprov" [(ngModel)]="draft.province_code" class="narrow" />
          </div>
          <div>
            <label>Model version *</label>
            <input name="cmodel" [(ngModel)]="draft.model_version" required />
          </div>
          <button type="submit" [disabled]="busy()">Insert read</button>
        </form>
        <p class="muted small">frame_ts is set to now on submit.</p>
      </details>
    }

    <p class="muted small">{{ reads().length }} row(s)</p>

    <table class="data">
      <thead>
        <tr>
          <th>Plate</th><th>Camera</th><th>Conf.</th><th>Frame time</th>
          <th>Model</th><th>Flags</th><th></th>
        </tr>
      </thead>
      <tbody>
        @for (r of reads(); track r.id) {
          <tr>
            <td class="mono strong">
              {{ r.plate_text }}
              @if (r.province_code) { <span class="muted">· {{ r.province_code }}</span> }
            </td>
            <td>{{ cameraName(r.camera_id) }}</td>
            <td>
              <span class="tag" [class.ok]="r.confidence >= 0.8"
                    [class.warn]="r.confidence < 0.8 && r.confidence >= 0.5"
                    [class.bad]="r.confidence < 0.5">
                {{ r.confidence | number: '1.2-2' }}
              </span>
            </td>
            <td class="mono small">{{ r.frame_ts | date: 'yyyy-MM-dd HH:mm:ss' }}</td>
            <td class="small">
              {{ r.model_version }}
              @if (!r.is_active_model) { <span class="tag off">stale</span> }
            </td>
            <td>
              @if (r.is_verified) { <span class="tag ok">verified</span> }
              @if (!r.is_valid) { <span class="tag bad">invalid</span> }
            </td>
            <td class="actions">
              @if (auth.canEdit() && !r.is_verified) {
                <button class="ghost" (click)="verify(r)">Verify</button>
              }
            </td>
          </tr>
        } @empty {
          <tr><td colspan="7" class="muted">No plate reads match these filters.</td></tr>
        }
      </tbody>
    </table>
  `,
})
export class PlateReadsComponent implements OnInit {
  private api = inject(ApiService);
  auth = inject(AuthService);

  reads = signal<PlateRead[]>([]);
  cameras = signal<Camera[]>([]);
  error = signal('');
  busy = signal(false);

  sinceLocal = '';
  untilLocal = '';
  q: PlateReadQuery = { only_active_model: true, limit: 100, offset: 0 };
  draft: PlateReadCreate = this.emptyDraft();

  ngOnInit(): void {
    this.api.listCameras().subscribe({
      next: (c) => this.cameras.set(c),
      error: () => {},
    });
    this.load();
  }

  load(): void {
    this.error.set('');
    const query: PlateReadQuery = { ...this.q };
    // datetime-local has no zone; convert to ISO so the API reads it correctly.
    if (this.sinceLocal) query.since = new Date(this.sinceLocal).toISOString();
    if (this.untilLocal) query.until = new Date(this.untilLocal).toISOString();
    this.api.listPlateReads(query).subscribe({
      next: (r) => this.reads.set(r),
      error: (e) => this.error.set(errorText(e)),
    });
  }

  reset(): void {
    this.q = { only_active_model: true, limit: 100, offset: 0 };
    this.sinceLocal = '';
    this.untilLocal = '';
    this.load();
  }

  create(): void {
    this.busy.set(true);
    this.error.set('');
    const body: PlateReadCreate = {
      ...this.draft,
      frame_ts: new Date().toISOString(),
      province_code: this.draft.province_code || null,
    };
    this.api.createPlateRead(body).subscribe({
      next: () => {
        this.busy.set(false);
        this.draft = this.emptyDraft();
        this.load();
      },
      error: (e) => {
        this.busy.set(false);
        this.error.set(errorText(e));
      },
    });
  }

  verify(r: PlateRead): void {
    this.api.verifyPlateRead(r.id).subscribe({
      next: () => this.load(),
      error: (e) => this.error.set(errorText(e)),
    });
  }

  cameraName(id: string): string {
    return this.cameras().find((c) => c.id === id)?.name ?? id.slice(0, 8);
  }

  private emptyDraft(): PlateReadCreate {
    return {
      camera_id: '',
      plate_text: '',
      confidence: 0.9,
      frame_ts: '',
      model_version: 'manual-test',
      province_code: '',
    };
  }
}
