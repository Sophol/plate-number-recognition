import { Component, inject, OnInit, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { Camera, ContainerLookup, ContainerRead, ContainerReadQuery } from '../../core/models';
import { errorText } from '../../core/error-text';

@Component({
  selector: 'app-containers',
  standalone: true,
  imports: [FormsModule, DatePipe, DecimalPipe],
  template: `
    <header class="page-head">
      <h2>Containers</h2>
      <button class="ghost" (click)="load()">Refresh</button>
    </header>

    @if (error()) { <div class="alert error">{{ error() }}</div> }

    <form class="card row-form" (ngSubmit)="lookup()">
      <div>
        <label>Check a container number</label>
        <input name="lookup" [(ngModel)]="lookupNumber" placeholder="e.g. CMAU 722427 0" class="mono" />
      </div>
      <button type="submit" [disabled]="!lookupNumber.trim()">Check</button>
      @if (info(); as i) {
        <div class="lookup-result">
          <span class="mono strong">{{ i.number }}</span>
          @if (!i.well_formed) { <span class="tag bad">not a container number</span> }
          @else {
            <span class="tag" [class.ok]="i.checksum_ok" [class.bad]="!i.checksum_ok">
              check digit {{ i.checksum_ok ? 'valid' : 'WRONG' }}
            </span>
            @if (i.is_known === true) { <span class="tag ok">known to PAS</span> }
            @if (i.is_known === false) { <span class="tag warn">not in PAS</span> }
            @if (i.is_known === null) { <span class="tag off">PAS list not loaded</span> }
            <span class="muted small">
              seen {{ i.read_count }} time(s)
              @if (i.last_seen) { · last {{ i.last_seen | date: 'yyyy-MM-dd HH:mm' }} }
            </span>
          }
        </div>
      }
    </form>

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
        <label>Number</label>
        <input name="num" [(ngModel)]="q.container_number" placeholder="partial match" />
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
      <button type="submit">Apply filters</button>
      <button type="button" class="ghost" (click)="reset()">Reset</button>
    </form>

    <p class="muted small">{{ reads().length }} row(s)</p>

    <table class="data">
      <thead>
        <tr>
          <th>Container</th><th>Owner</th><th>Camera</th><th>Conf.</th>
          <th>Frame time</th><th>Model</th><th>Flags</th>
        </tr>
      </thead>
      <tbody>
        @for (r of reads(); track r.id) {
          <tr>
            <td class="mono strong">
              {{ r.container_number }}
              @if (r.was_snapped) {
                <span class="muted small" title="OCR read {{ r.ocr_text }}; corrected against the PAS list">· corrected</span>
              }
            </td>
            <td class="mono">{{ r.owner_code }}</td>
            <td>{{ cameraName(r.camera_id) }}</td>
            <td>
              <span class="tag" [class.ok]="r.confidence >= 0.8"
                    [class.warn]="r.confidence < 0.8 && r.confidence >= 0.5"
                    [class.bad]="r.confidence < 0.5">
                {{ r.confidence | number: '1.2-2' }}
              </span>
            </td>
            <td class="mono small">{{ r.frame_ts | date: 'yyyy-MM-dd HH:mm:ss' }}</td>
            <td class="small">{{ r.model_version }}</td>
            <td>
              @if (r.checksum_ok) { <span class="tag ok">checksum</span> }
              @else { <span class="tag bad">bad check digit</span> }
              @if (r.is_known === true) { <span class="tag ok">known</span> }
              @if (r.is_known === false) { <span class="tag warn">unknown</span> }
            </td>
          </tr>
        } @empty {
          <tr><td colspan="7" class="muted">No container reads match these filters.</td></tr>
        }
      </tbody>
    </table>
  `,
  styles: [`
    .lookup-result { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }
  `],
})
export class ContainersComponent implements OnInit {
  private api = inject(ApiService);

  reads = signal<ContainerRead[]>([]);
  cameras = signal<Camera[]>([]);
  info = signal<ContainerLookup | null>(null);
  error = signal('');

  lookupNumber = '';
  sinceLocal = '';
  untilLocal = '';
  q: ContainerReadQuery = { limit: 100, offset: 0 };

  ngOnInit(): void {
    this.api.listCameras().subscribe({ next: (c) => this.cameras.set(c), error: () => {} });
    this.load();
  }

  load(): void {
    this.error.set('');
    const query: ContainerReadQuery = { ...this.q };
    if (this.sinceLocal) query.since = new Date(this.sinceLocal).toISOString();
    if (this.untilLocal) query.until = new Date(this.untilLocal).toISOString();
    this.api.listContainerReads(query).subscribe({
      next: (r) => this.reads.set(r),
      error: (e) => this.error.set(errorText(e)),
    });
  }

  lookup(): void {
    this.error.set('');
    this.api.lookupContainer(this.lookupNumber).subscribe({
      next: (i) => this.info.set(i),
      error: (e) => this.error.set(errorText(e)),
    });
  }

  reset(): void {
    this.q = { limit: 100, offset: 0 };
    this.sinceLocal = '';
    this.untilLocal = '';
    this.load();
  }

  cameraName(id: string): string {
    return this.cameras().find((c) => c.id === id)?.name ?? id.slice(0, 8);
  }
}
