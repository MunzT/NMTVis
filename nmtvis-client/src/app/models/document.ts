export class Document {
    constructor(readonly id: string, readonly name: string, readonly sentences: any) {
    }
}

export class Sentence {
    constructor(readonly id: string, readonly inputSentence: string, readonly translation: string,
                readonly attention: number[][], readonly beam: any) {
    }
}
