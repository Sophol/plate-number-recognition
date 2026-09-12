import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { ApiService } from '../../core/api.service';
import { AuthService } from '../../core/auth.service';
import { LiveCamera, StageEvent } from '../../core/models';
import { errorText } from '../../core/error-text';

const EVENT_POLL_MS = 1000;

@Component({
  selector: 'app-live',
  standalone: true,
  imports: [DatePipe, DecimalPipe],
  templateUrl: './live.component.html',
  styleUrl: './live.component.css',
})
export class LiveComponent implements OnInit, OnDestroy {
  private api = inject(ApiService);
  private auth = inject(AuthService);
  private sanitizer = inject(DomSanitizer);

  cameras = signal<LiveCamera[]>([]);
  selected = signal<LiveCamera | null>(null);
  streamUrl = signal<SafeUrl | null>(null);
  events = signal<StageEvent[]>([]);
  fps = signal(0);
  connected = signal(false);
  detail = signal('');
  error = signal('');
  streamFailed = signal(false);

  private timer: ReturnType<typeof setInterval> | null = null;

  ngOnInit(): void {
    this.api.liveCameras().subscribe({
      next: (cams) => {
        this.cameras.set(cams);
        if (cams.length) this.select(cams[0]);
      },
      error: (e) => this.error.set(errorText(e)),
    });
  }

  ngOnDestroy(): void {
    this.stopPolling();
  }

  select(camera: LiveCamera): void {
    this.stopPolling();
    this.selected.set(camera);
    this.events.set([]);
    this.streamFailed.set(false);
    this.connected.set(false);
    this.fps.set(0);
    this.detail.set('');

    const token = this.auth.token;
    if (!token) return;

    // Angular blocks a raw URL in [src]; the token makes it look unsafe.
    this.streamUrl.set(
      this.sanitizer.bypassSecurityTrustUrl(this.api.liveStreamUrl(camera.id, token)),
    );
    this.timer = setInterval(() => this.pollEvents(), EVENT_POLL_MS);
    this.pollEvents();
  }

  onStreamError(): void {
    this.streamFailed.set(true);
  }

  private pollEvents(): void {
    const camera = this.selected();
    if (!camera) return;
    this.api.liveEvents(camera.id, 25).subscribe({
      next: (res) => {
        this.events.set(res.events);
        this.connected.set(res.status?.connected ?? false);
        this.fps.set(res.status?.fps ?? 0);
        this.detail.set(res.status?.detail ?? '');
      },
      error: (e) => this.error.set(errorText(e)),
    });
  }

  private stopPolling(): void {
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    this.streamUrl.set(null);
  }

  /** Cameras saved without a name would otherwise render as a blank tab. */
  label(camera: LiveCamera): string {
    return camera.name?.trim() || `camera ${camera.id.slice(0, 8)}`;
  }

  /** Min of detector and OCR, matching what the pipeline stores as confidence. */
  overall(e: StageEvent): number {
    return Math.min(e.detector_confidence ?? 0, e.ocr_confidence ?? 0);
  }

  barClass(value: number): string {
    if (value >= 0.8) return 'ok';
    if (value >= 0.5) return 'warn';
    return 'bad';
  }
}
