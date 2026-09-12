import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { API_BASE } from './api-base';
import {
  Camera,
  CameraCreate,
  CameraUpdate,
  ListType,
  PlateRead,
  PlateReadCreate,
  PlateReadQuery,
  LiveCamera,
  LiveEvents,
  Vehicle,
  VehicleCreate,
  VehicleUpdate,
  ContainerLookup,
  ContainerRead,
  ContainerReadQuery,
} from './models';

/** Drops undefined/null/'' so we never send empty filters as literal strings. */
function params(obj: Record<string, unknown>): HttpParams {
  let p = new HttpParams();
  for (const [k, v] of Object.entries(obj)) {
    if (v === undefined || v === null || v === '') continue;
    p = p.set(k, String(v));
  }
  return p;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  constructor(private http: HttpClient) {}

  health(): Observable<unknown> {
    return this.http.get(`${API_BASE}/health`);
  }

  // --- cameras ---
  listCameras(isActive?: boolean): Observable<Camera[]> {
    return this.http.get<Camera[]>(`${API_BASE}/cameras`, {
      params: params({ is_active: isActive }),
    });
  }

  getCamera(id: string): Observable<Camera> {
    return this.http.get<Camera>(`${API_BASE}/cameras/${id}`);
  }

  createCamera(body: CameraCreate): Observable<Camera> {
    return this.http.post<Camera>(`${API_BASE}/cameras`, body);
  }

  updateCamera(id: string, body: CameraUpdate): Observable<Camera> {
    return this.http.patch<Camera>(`${API_BASE}/cameras/${id}`, body);
  }

  // --- plate reads ---
  listPlateReads(q: PlateReadQuery = {}): Observable<PlateRead[]> {
    return this.http.get<PlateRead[]>(`${API_BASE}/plate-reads`, {
      params: params(q as Record<string, unknown>),
    });
  }

  createPlateRead(body: PlateReadCreate): Observable<PlateRead> {
    return this.http.post<PlateRead>(`${API_BASE}/plate-reads`, body);
  }

  verifyPlateRead(id: string): Observable<PlateRead> {
    return this.http.post<PlateRead>(`${API_BASE}/plate-reads/${id}/verify`, {});
  }

  // --- containers ---
  listContainerReads(q: ContainerReadQuery = {}): Observable<ContainerRead[]> {
    return this.http.get<ContainerRead[]>(`${API_BASE}/container-reads`, {
      params: params(q as Record<string, unknown>),
    });
  }

  lookupContainer(number: string): Observable<ContainerLookup> {
    return this.http.get<ContainerLookup>(`${API_BASE}/containers/${encodeURIComponent(number.trim())}`);
  }

  // --- vehicles ---
  listVehicles(opts: {
    list_type?: ListType;
    plate_text?: string;
    limit?: number;
    offset?: number;
  } = {}): Observable<Vehicle[]> {
    return this.http.get<Vehicle[]>(`${API_BASE}/vehicles`, {
      params: params(opts as Record<string, unknown>),
    });
  }

  getVehicle(id: string): Observable<Vehicle> {
    return this.http.get<Vehicle>(`${API_BASE}/vehicles/${id}`);
  }

  createVehicle(body: VehicleCreate): Observable<Vehicle> {
    return this.http.post<Vehicle>(`${API_BASE}/vehicles`, body);
  }

  updateVehicle(id: string, body: VehicleUpdate): Observable<Vehicle> {
    return this.http.patch<Vehicle>(`${API_BASE}/vehicles/${id}`, body);
  }

  deleteVehicle(id: string): Observable<void> {
    return this.http.delete<void>(`${API_BASE}/vehicles/${id}`);
  }

  // --- live preview ---
  liveCameras(): Observable<LiveCamera[]> {
    return this.http.get<LiveCamera[]>(`${API_BASE}/live/cameras`);
  }

  liveEvents(cameraId: string, limit = 20): Observable<LiveEvents> {
    return this.http.get<LiveEvents>(`${API_BASE}/live/${cameraId}/events`, {
      params: params({ limit }),
    });
  }

  /** MJPEG URL for an <img> tag. The token rides in the query string because
   *  an image request cannot carry an Authorization header. */
  liveStreamUrl(cameraId: string, token: string): string {
    return `${API_BASE}/live/${cameraId}/stream?token=${encodeURIComponent(token)}`;
  }
}
