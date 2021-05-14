import {Injectable} from '@angular/core';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import {Observable} from 'rxjs';

@Injectable()
export class ExperimentService {

    readonly EXPERIMENT_URL: string = "http://localhost:5000/api/experiments";

    constructor(private http: HttpClient) {
    }

    sendSurveyData(data) {
        return this.http.post(this.EXPERIMENT_URL + "/surveydata", data);
    }

    getNextSentence(experimentMetrics) {
        return this.http.post(this.EXPERIMENT_URL + "/next", {"experimentMetrics": experimentMetrics});
    }

    getExperimentData() {
        return this.http.get(this.EXPERIMENT_URL + "/experimentdata");
    }

}
