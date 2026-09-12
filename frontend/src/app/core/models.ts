export type ListType = 'whitelist' | 'blacklist' | 'neutral';
export type Role = 'viewer' | 'list_editor' | 'gate_operator' | 'admin';

export const ROLE_RANK: Record<Role, number> = {
  viewer: 0,
  list_editor: 1,
  gate_operator: 2,
  admin: 3,
};

export interface Token {
  access_token: string;
  token_type?: string;
}

export interface Me {
  username: string;
  role: Role;
}

export interface Camera {
  id: string;
  name: string;
  rtsp_url: string;
  site_id: string | null;
  direction: string | null;
  is_active: boolean;
  created_at: string;
}

export interface CameraCreate {
  name: string;
  rtsp_url: string;
  site_id?: string | null;
  direction?: string | null;
  is_active?: boolean;
}

export type CameraUpdate = Partial<CameraCreate>;

export interface PlateRead {
  id: string;
  camera_id: string;
  track_id: string | null;
  plate_text: string;
  province_code: string | null;
  plate_type: string | null;
  vehicle_type: string | null;
  confidence: number;
  detector_confidence: number | null;
  ocr_confidence: number | null;
  province_confidence: number | null;
  frame_ts: string;
  image_path: string | null;
  plate_crop_path: string | null;
  model_version: string;
  is_active_model: boolean;
  is_valid: boolean;
  is_verified: boolean;
  created_at: string;
}

export interface PlateReadCreate {
  camera_id: string;
  plate_text: string;
  confidence: number;
  frame_ts: string;
  model_version: string;
  track_id?: string | null;
  province_code?: string | null;
  plate_type?: string | null;
  vehicle_type?: string | null;
  detector_confidence?: number | null;
  ocr_confidence?: number | null;
  province_confidence?: number | null;
  image_path?: string | null;
  plate_crop_path?: string | null;
  is_active_model?: boolean;
  is_valid?: boolean;
}

export interface PlateReadQuery {
  camera_id?: string;
  plate_text?: string;
  since?: string;
  until?: string;
  only_active_model?: boolean;
  limit?: number;
  offset?: number;
}

export interface Vehicle {
  id: string;
  plate_text: string;
  owner_name: string | null;
  phone: string | null;
  list_type: ListType;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface VehicleCreate {
  plate_text: string;
  owner_name?: string | null;
  phone?: string | null;
  list_type?: ListType;
  notes?: string | null;
}

export interface VehicleUpdate {
  owner_name?: string | null;
  phone?: string | null;
  list_type?: ListType | null;
  notes?: string | null;
  reason?: string | null;
}

export interface PreviewStatus {
  camera_id: string;
  name: string;
  connected: boolean;
  fps: number;
  viewers: number;
  detail: string;
  attempts: number;
}

export interface LiveCamera {
  id: string;
  name: string;
  rtsp_url: string;
  preview: PreviewStatus | null;
}

/** One detection's trip through the pipeline, as the preview observed it. */
export interface StageEvent {
  camera_id: string;
  at: string;
  detector_confidence: number;
  ocr_text: string;
  ocr_confidence: number;
  corrected_text: string;
  province_code: string | null;
  province_confidence: number | null;
  plate_type: string;
  is_valid: boolean;
  warped: boolean;
}

export interface LiveEvents {
  status: PreviewStatus | null;
  events: StageEvent[];
}
