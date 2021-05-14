import {Component} from '@angular/core';
import {HttpClient} from '@angular/common/http';
import * as d3 from 'd3';
import {AuthService} from './services/auth.service';
import {Router} from '@angular/router';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})

export class AppComponent {
    title = 'NMTVis';

    constructor(private http: HttpClient, private auth: AuthService, private router: Router) {
    }

    logout() {
        this.router.navigate(['/start']);
    }

    ngAfterContentInit() {

    }
}
