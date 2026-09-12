import { Component, inject, OnInit, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { AuthService } from '../../core/auth.service';
import { ListType, Vehicle, VehicleCreate } from '../../core/models';
import { errorText } from '../../core/error-text';

const LIST_TYPES: ListType[] = ['whitelist', 'blacklist', 'neutral'];

@Component({
  selector: 'app-vehicles',
  standalone: true,
  imports: [FormsModule, DatePipe],
  template: `
    <header class="page-head">
      <h2>Vehicles</h2>
      <button class="ghost" (click)="load()">Refresh</button>
    </header>

    @if (error()) { <div class="alert error">{{ error() }}</div> }

    <form class="card row-form" (ngSubmit)="load()">
      <div>
        <label>List type</label>
        <select name="lt" [(ngModel)]="filterListType">
          <option value="">any</option>
          @for (t of listTypes; track t) { <option [value]="t">{{ t }}</option> }
        </select>
      </div>
      <div>
        <label>Plate text</label>
        <input name="pt" [(ngModel)]="filterPlate" placeholder="partial match" />
      </div>
      <button type="submit">Search</button>
    </form>

    @if (auth.canEdit()) {
      <form class="card row-form" (ngSubmit)="create()">
        <div>
          <label>Plate text *</label>
          <input name="plate" [(ngModel)]="draft.plate_text" required />
        </div>
        <div>
          <label>Owner</label>
          <input name="owner" [(ngModel)]="draft.owner_name" />
        </div>
        <div>
          <label>Phone</label>
          <input name="phone" [(ngModel)]="draft.phone" />
        </div>
        <div>
          <label>List type</label>
          <select name="dlt" [(ngModel)]="draft.list_type">
            @for (t of listTypes; track t) { <option [value]="t">{{ t }}</option> }
          </select>
        </div>
        <div class="grow">
          <label>Notes</label>
          <input name="notes" [(ngModel)]="draft.notes" />
        </div>
        <button type="submit" [disabled]="busy()">Add vehicle</button>
      </form>
    } @else {
      <div class="alert info">Your role is read-only. Editing the vehicle list needs list_editor or higher.</div>
    }

    <table class="data">
      <thead>
        <tr><th>Plate</th><th>Owner</th><th>Phone</th><th>List</th><th>Notes</th><th>Updated</th><th></th></tr>
      </thead>
      <tbody>
        @for (v of vehicles(); track v.id) {
          <tr>
            <td class="mono strong">{{ v.plate_text }}</td>
            <td>{{ v.owner_name || '—' }}</td>
            <td>{{ v.phone || '—' }}</td>
            <td>
              @if (auth.canEdit()) {
                <select [ngModel]="v.list_type" (ngModelChange)="changeList(v, $event)"
                        [ngModelOptions]="{standalone: true}">
                  @for (t of listTypes; track t) { <option [value]="t">{{ t }}</option> }
                </select>
              } @else {
                <span class="tag" [class.ok]="v.list_type === 'whitelist'"
                      [class.bad]="v.list_type === 'blacklist'">{{ v.list_type }}</span>
              }
            </td>
            <td class="small">{{ v.notes || '—' }}</td>
            <td class="mono small">{{ v.updated_at | date: 'yyyy-MM-dd HH:mm' }}</td>
            <td class="actions">
              @if (auth.canEdit()) {
                <button class="ghost danger" (click)="remove(v)">Delete</button>
              }
            </td>
          </tr>
        } @empty {
          <tr><td colspan="7" class="muted">No vehicles.</td></tr>
        }
      </tbody>
    </table>
  `,
})
export class VehiclesComponent implements OnInit {
  private api = inject(ApiService);
  auth = inject(AuthService);

  vehicles = signal<Vehicle[]>([]);
  error = signal('');
  busy = signal(false);
  listTypes = LIST_TYPES;

  filterListType: ListType | '' = '';
  filterPlate = '';
  draft: VehicleCreate = this.emptyDraft();

  ngOnInit(): void {
    this.load();
  }

  load(): void {
    this.error.set('');
    this.api
      .listVehicles({
        list_type: this.filterListType || undefined,
        plate_text: this.filterPlate || undefined,
      })
      .subscribe({
        next: (v) => this.vehicles.set(v),
        error: (e) => this.error.set(errorText(e)),
      });
  }

  create(): void {
    this.busy.set(true);
    this.error.set('');
    this.api
      .createVehicle({
        plate_text: this.draft.plate_text,
        owner_name: this.draft.owner_name || null,
        phone: this.draft.phone || null,
        list_type: this.draft.list_type,
        notes: this.draft.notes || null,
      })
      .subscribe({
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

  changeList(v: Vehicle, list_type: ListType): void {
    this.api
      .updateVehicle(v.id, { list_type, reason: 'changed from test console' })
      .subscribe({
        next: () => this.load(),
        error: (e) => {
          this.error.set(errorText(e));
          this.load();
        },
      });
  }

  remove(v: Vehicle): void {
    if (!confirm(`Delete vehicle ${v.plate_text}?`)) return;
    this.api.deleteVehicle(v.id).subscribe({
      next: () => this.load(),
      error: (e) => this.error.set(errorText(e)),
    });
  }

  private emptyDraft(): VehicleCreate {
    return {
      plate_text: '',
      owner_name: '',
      phone: '',
      list_type: 'neutral',
      notes: '',
    };
  }
}
