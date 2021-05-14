import {BrowserModule} from '@angular/platform-browser';
import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import {FormsModule}   from '@angular/forms';
import {HttpClientModule} from '@angular/common/http';
import {HTTP_INTERCEPTORS} from '@angular/common/http';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {MatButtonModule} from '@angular/material/button';
import {MatToolbarModule} from '@angular/material/toolbar';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatIconModule} from '@angular/material/icon';
import {MatCardModule} from '@angular/material/card';
import {MatSidenavModule} from '@angular/material/sidenav';
import {MatTableModule} from '@angular/material/table';
import {MatListModule} from '@angular/material/list';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatDividerModule} from '@angular/material/divider';
import {MatSliderModule} from '@angular/material/slider';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatSnackBarModule} from '@angular/material/snack-bar';
import {MatInputModule} from '@angular/material/input';
import {MatGridListModule} from '@angular/material/grid-list';
import {MatDialogModule} from '@angular/material/dialog';
import {MatFileUploadModule} from 'angular-material-fileupload';
import {AppComponent} from './app.component';
import {DocumentsOverviewComponent, DocumentUploadDialog} from './documents-overview/documents-overview.component';
import {SentenceViewComponent} from './sentence-view/sentence-view.component';
import {BeamNodeDialog, InfoDialog} from './sentence-view/sentence-view.component';
import {DocumentService} from './services/document.service';
import {SentencesVisComponent} from './documents-overview/sentences-vis/sentences-vis.component';
import {LoginComponent} from './login/login.component';
import {AuthService} from './services/auth.service';
import {EnsureAuthenticated} from './services/ensure-authenticated.service';
import {LoggedinRedirect} from './services/loggedin-redirect.service';
import {ExperimentService} from './services/experiment.service';
import {TokenInterceptor} from './auth/token-interceptor';
import {JwtInterceptor} from './auth/jwt-interceptor';
import {RegisterComponent} from './register/register.component';
import {TextDisplayPipe} from './pipes/text-display.pipe';
import {ParallelCoordinatesComponent} from './documents-overview/parallel-coordinates/parallel-coordinates.component';
import {SentenceListItemComponent} from './documents-overview/sentence-list-item/sentence-list-item.component';
import {IntroComponent} from './intro/intro.component';

const appRoutes: Routes = [
    {path: 'login', component: LoginComponent, canActivate: [LoggedinRedirect]},
    {path: 'intro', component: IntroComponent, canActivate: [EnsureAuthenticated]},
    {path: 'register', component: RegisterComponent},
    {path: 'documents', component: DocumentsOverviewComponent, canActivate: [EnsureAuthenticated]},
    {
        path: 'documents/:document_id/sentence/:sentence_id',
        component: DocumentsOverviewComponent,
        canActivate: [EnsureAuthenticated]
    },
    {
        path: 'document/:document_id/sentence/:sentence_id',
        component: SentenceViewComponent,
        canActivate: [EnsureAuthenticated]
    },
    {
        path: 'beam/document/:document_id/sentence/:sentence_id',
        component: SentenceViewComponent,
        canActivate: [EnsureAuthenticated]
    },
];

@NgModule({
    declarations: [
        AppComponent,
        DocumentsOverviewComponent, LoginComponent, InfoDialog,
        SentenceViewComponent, BeamNodeDialog, SentencesVisComponent,
        DocumentUploadDialog, RegisterComponent,
        TextDisplayPipe,
        ParallelCoordinatesComponent, SentenceListItemComponent, IntroComponent
    ],
    imports: [
        BrowserModule, HttpClientModule, FormsModule, BrowserAnimationsModule, MatSnackBarModule,
        MatButtonModule, MatToolbarModule, MatInputModule, MatProgressSpinnerModule, MatCheckboxModule,
        MatIconModule, MatCardModule, MatTableModule, MatSidenavModule, MatListModule, MatDividerModule,
        MatSliderModule, MatInputModule, MatGridListModule, MatDialogModule, MatFileUploadModule, MatProgressBarModule,
        RouterModule.forRoot(
            appRoutes, // <-- debugging purposes only
        )
    ],
    entryComponents: [
        BeamNodeDialog, DocumentUploadDialog, InfoDialog
    ],
    providers: [DocumentService, AuthService, TextDisplayPipe, EnsureAuthenticated, LoggedinRedirect, ExperimentService, {
        provide: HTTP_INTERCEPTORS,
        useClass: TokenInterceptor,
        multi: true
    }, {
        provide: HTTP_INTERCEPTORS,
        useClass: JwtInterceptor,
        multi: true
    }],
    bootstrap: [AppComponent]
})
export class AppModule {
}
