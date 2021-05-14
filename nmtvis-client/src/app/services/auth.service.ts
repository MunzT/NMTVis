import {Injectable} from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import {Observable} from 'rxjs';

@Injectable()
export class AuthService {

    private BASE_URL: string = 'http://localhost:5000/api/auth';

    constructor(private http: HttpClient) {
    }

    login(user): Observable<any> {
        return this.http.post(this.BASE_URL + "/login", user);
    }

    logout() {
        localStorage.clear();
    }

    register(user): Observable<any> {
        return this.http.post(this.BASE_URL + "/register", user);
    }

    getToken(): string {
        let token = localStorage.getItem("access_token");
        return token ? token : "eyJhbGciOiJIUzI1NiJ9.eyJ0ZXN0IjoidGVzdCJ9.yu6oYJHutVjzwVtZ5bs4CTVhFUcEhFS5-QvvMyBfDLY";
    }

}
