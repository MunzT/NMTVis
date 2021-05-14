import {Injectable} from '@angular/core';
import {Document, Sentence} from '../models/document';
import {HttpClient, HttpHeaders} from '@angular/common/http';
import {Observable} from 'rxjs';

@Injectable()
export class DocumentService {

    private documents: Document[] = [new Document("1", "Dokument 1", "Das ist der Inhalt.ich bin nett."),
        new Document("2", "Dokument 2", "Ein Satz.")];
    private url = "http://localhost:5000";
    private documentsUrl = this.url + "/api/documents";

    constructor(private http: HttpClient) {
    }

    getSentences(document_id): Observable<Sentence[]> {
        return this.http.get<Sentence[]>(this.documentsUrl + "/" + document_id + "/sentences");
    }

    getDocuments(): Observable<Document[]> {
        return this.http.get<Document[]>(this.documentsUrl);
    }

    getSentence(document_id, sentence_id): Observable<Sentence> {
        return this.http.get<Sentence>(this.documentsUrl + "/" + document_id + "/sentences/" + sentence_id);
    }

    setCorrected(document_id: string, sentence_id: string, corrected: boolean) {
        return this.http.post(this.documentsUrl + "/" + document_id + "/sentences/" + sentence_id + "/corrected",
            {"corrected": corrected});
    }

    setFlagged(document_id: string, sentence_id: string, flagged: boolean) {
        return this.http.post(this.documentsUrl + "/" + document_id + "/sentences/" + sentence_id + "/flagged",
            {"flagged": flagged});
    }

    retrain(document_id: string) {
        return this.http.post(this.documentsUrl + "/" + document_id + "/retrain", {});
    }

    retranslate(document_id: string) {
        return this.http.post(this.documentsUrl + "/" + document_id + "/translate", {});
    }

    retranslateSentence(document_id: string, sentence_id: string, beam_size: number) {
        return this.http.post(this.documentsUrl + "/" + document_id + "/sentences/" + sentence_id + "/beam_size/" + beam_size + "/translateSentence", {});
    }

    saveTranslation(document_id: string) {
        return this.http.post(this.documentsUrl + "/" + document_id + "/saveTranslation", {});
    }

    filterForSimilarSentences(document_id: string, sentence_id: string) {
        return this.http.post(this.documentsUrl + "/" + document_id + "/filterForSimilarSentences" + "/" + sentence_id, {});
    }
}
