import { Routes } from '@angular/router';
import { authGuard } from './core/auth.guard';
import { LoginComponent } from './pages/login/login.component';
import { CamerasComponent } from './pages/cameras/cameras.component';
import { PlateReadsComponent } from './pages/plate-reads/plate-reads.component';
import { VehiclesComponent } from './pages/vehicles/vehicles.component';
import { LiveComponent } from './pages/live/live.component';

export const routes: Routes = [
  { path: 'login', component: LoginComponent },
  { path: 'live', component: LiveComponent, canActivate: [authGuard] },
  { path: 'cameras', component: CamerasComponent, canActivate: [authGuard] },
  { path: 'plate-reads', component: PlateReadsComponent, canActivate: [authGuard] },
  { path: 'vehicles', component: VehiclesComponent, canActivate: [authGuard] },
  { path: '', pathMatch: 'full', redirectTo: 'plate-reads' },
  { path: '**', redirectTo: 'plate-reads' },
];
